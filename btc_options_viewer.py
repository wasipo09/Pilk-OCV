#!/usr/bin/env python3
"""
BTC Options Viewer - Fetches options data from Deribit and Binance (no API key required)
Shows Greeks (Delta, Gamma, Theta, Vega, Rho) and Put/Call prices at each strike

Usage:
    python btc_options_viewer.py                  # Interactive menu
    python btc_options_viewer.py --dump-binance   # Export Binance options (7 days)
    python btc_options_viewer.py --dump-deribit   # Export Deribit options (7 days)
"""

import argparse
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# API Base URLs
DERIBIT_API = "https://www.deribit.com/api/v2"
BINANCE_API = "https://eapi.binance.com"


# =============================================================================
# CCXT SPOT PRICE FETCHING
# =============================================================================

def fetch_live_spot_price() -> float:
    """Get current BTC spot price using CCXT (Binance exchange)."""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        return ticker['last']
    except Exception as e:
        raise RuntimeError(f"Failed to fetch live spot price from CCXT: {e}")

# CSV column order
CSV_COLUMNS = [
    "exchange", "expiry", "strike", "underlying_price",
    "call_bid_btc", "call_ask_btc", "call_mark_btc",
    "call_bid_usd", "call_ask_usd", "call_mark_usd",
    "call_iv", "call_delta", "call_gamma", "call_theta", "call_vega", "call_rho",
    "put_bid_btc", "put_ask_btc", "put_mark_btc",
    "put_bid_usd", "put_ask_usd", "put_mark_usd",
    "put_iv", "put_delta", "put_gamma", "put_theta", "put_vega", "put_rho"
]


# =============================================================================
# API HELPERS
# =============================================================================

def api_get(url: str, params: dict = None) -> dict:
    """Make GET request with error handling"""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# =============================================================================
# DERIBIT FUNCTIONS
# =============================================================================

def deribit_get_index_price() -> float:
    """Get current BTC index price from Deribit (for comparison, not used for main analysis)."""
    data = api_get(f"{DERIBIT_API}/public/get_index_price", {"index_name": "btc_usd"})
    return data["result"]["index_price"]


def deribit_get_instruments() -> list:
    """Get all BTC options instruments from Deribit"""
    data = api_get(f"{DERIBIT_API}/public/get_instruments",
                   {"currency": "BTC", "kind": "option", "expired": "false"})
    return data["result"]


def deribit_get_ticker(instrument_name: str) -> dict:
    """Get ticker data for a specific instrument"""
    data = api_get(f"{DERIBIT_API}/public/ticker", {"instrument_name": instrument_name})
    return data["result"]


def deribit_parse_instrument(name: str) -> tuple:
    """Parse Deribit instrument name, returns (expiry, strike, type)"""
    parts = name.split("-")
    return parts[1], int(parts[2]), "call" if parts[3] == "C" else "put"


def deribit_expiry_to_date(expiry: str) -> datetime:
    """Convert Deribit expiry string to datetime"""
    return datetime.strptime(expiry, "%d%b%y")


def deribit_get_expiration_dates() -> list:
    """Get sorted unique expiration dates from Deribit"""
    instruments = deribit_get_instruments()
    expirations = {deribit_parse_instrument(i["instrument_name"])[0] for i in instruments}
    return sorted(expirations, key=deribit_expiry_to_date)


def deribit_get_expirations_within_days(days: int = 7) -> list:
    """Get expiration dates within N days from now"""
    cutoff = datetime.now() + timedelta(days=days)
    return [e for e in deribit_get_expiration_dates() if deribit_expiry_to_date(e) <= cutoff]


def deribit_fetch_single_ticker(inst: dict) -> tuple:
    """Fetch single ticker - used for parallel execution"""
    try:
        ticker = deribit_get_ticker(inst["instrument_name"])
        return inst["instrument_name"], ticker
    except Exception as e:
        return inst["instrument_name"], None


def deribit_get_options_chain(expiry: str, progress_callback=None, use_parallel: bool = True) -> pd.DataFrame:
    """Get complete options chain for a specific expiration date from Deribit"""
    instruments = [i for i in deribit_get_instruments() if f"-{expiry}-" in i["instrument_name"]]

    if not instruments:
        return pd.DataFrame()

    chain_data = {}
    total = len(instruments)
    completed = 0

    def process_ticker(name: str, ticker: dict):
        if not ticker:
            return
        expiry_str, strike, opt_type = deribit_parse_instrument(name)

        if strike not in chain_data:
            chain_data[strike] = {"strike": strike, "expiry": expiry_str}

        greeks = ticker.get("greeks") or {}
        underlying = ticker.get("underlying_price", 0) or 0
        bid = ticker.get("best_bid_price", 0) or 0
        ask = ticker.get("best_ask_price", 0) or 0
        mark = ticker.get("mark_price", 0) or 0

        chain_data[strike].update({
            f"{opt_type}_bid_btc": bid,
            f"{opt_type}_ask_btc": ask,
            f"{opt_type}_mark_btc": mark,
            f"{opt_type}_bid_usd": bid * underlying,
            f"{opt_type}_ask_usd": ask * underlying,
            f"{opt_type}_mark_usd": mark * underlying,
            f"{opt_type}_iv": ticker.get("mark_iv", 0) or 0,
            f"{opt_type}_delta": greeks.get("delta", 0) or 0,
            f"{opt_type}_gamma": greeks.get("gamma", 0) or 0,
            f"{opt_type}_theta": greeks.get("theta", 0) or 0,
            f"{opt_type}_vega": greeks.get("vega", 0) or 0,
            f"{opt_type}_rho": greeks.get("rho", 0) or 0,
            f"{opt_type}_oi": ticker.get("open_interest", 0) or 0,
            "underlying_price": underlying
        })

    if use_parallel:
        # Parallel fetching with rate limiting
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(deribit_fetch_single_ticker, inst): inst for inst in instruments}
            for future in as_completed(futures):
                name, ticker = future.result()
                process_ticker(name, ticker)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, name)
                time.sleep(0.02)  # Rate limiting
    else:
        # Sequential fetching
        for inst in instruments:
            try:
                ticker = deribit_get_ticker(inst["instrument_name"])
                process_ticker(inst["instrument_name"], ticker)
            except Exception:
                pass
            completed += 1
            if progress_callback:
                progress_callback(completed, total, inst["instrument_name"])
            time.sleep(0.05)

    df = pd.DataFrame(list(chain_data.values()))
    return df.sort_values("strike").reset_index(drop=True) if not df.empty else df


# =============================================================================
# BINANCE FUNCTIONS
# =============================================================================

def binance_get_index_price() -> float:
    """Get current BTC index price from Binance"""
    data = api_get(f"{BINANCE_API}/eapi/v1/index", {"underlying": "BTCUSDT"})
    return float(data["indexPrice"])


def binance_get_exchange_info() -> list:
    """Get all BTC options instruments from Binance"""
    data = api_get(f"{BINANCE_API}/eapi/v1/exchangeInfo")
    return [s for s in data["optionSymbols"] if s["symbol"].startswith("BTC")]


def binance_get_all_tickers() -> dict:
    """Get 24hr ticker data for all options as dict"""
    data = api_get(f"{BINANCE_API}/eapi/v1/ticker")
    return {t["symbol"]: t for t in data}


def binance_get_all_marks() -> dict:
    """Get mark prices with Greeks for all options as dict"""
    data = api_get(f"{BINANCE_API}/eapi/v1/mark")
    return {m["symbol"]: m for m in data}


def binance_parse_instrument(symbol: str) -> tuple:
    """Parse Binance instrument name, returns (expiry, strike, type)"""
    parts = symbol.split("-")
    return parts[1], int(parts[2]), "call" if parts[3] == "C" else "put"


def binance_format_expiry(expiry: str) -> str:
    """Convert YYMMDD to readable format (e.g., 02FEB26)"""
    return datetime.strptime(expiry, "%y%m%d").strftime("%d%b%y").upper()


def binance_expiry_to_date(expiry: str) -> datetime:
    """Convert Binance expiry string to datetime"""
    return datetime.strptime(expiry, "%y%m%d")


def binance_get_expiration_dates() -> list:
    """Get sorted unique expiration dates from Binance"""
    expirations = {binance_parse_instrument(s["symbol"])[0] for s in binance_get_exchange_info()}
    return sorted(expirations)


def binance_get_expirations_within_days(days: int = 7) -> list:
    """Get expiration dates within N days from now"""
    cutoff = datetime.now() + timedelta(days=days)
    return [e for e in binance_get_expiration_dates() if binance_expiry_to_date(e) <= cutoff]


def binance_get_options_chain(expiry: str, progress_callback=None) -> pd.DataFrame:
    """Get complete options chain for a specific expiration date from Binance"""
    # Bulk fetch all data (fast)
    tickers = binance_get_all_tickers()
    marks = binance_get_all_marks()
    underlying_price = binance_get_index_price()

    symbols = [s["symbol"] for s in binance_get_exchange_info() if f"-{expiry}-" in s["symbol"]]

    if not symbols:
        return pd.DataFrame()

    chain_data = {}

    for idx, symbol in enumerate(symbols):
        _, strike, opt_type = binance_parse_instrument(symbol)

        if strike not in chain_data:
            chain_data[strike] = {"strike": strike, "expiry": binance_format_expiry(expiry)}

        ticker = tickers.get(symbol, {})
        mark = marks.get(symbol, {})

        bid_usd = float(ticker.get("bidPrice") or 0)
        ask_usd = float(ticker.get("askPrice") or 0)
        mark_usd = float(mark.get("markPrice") or 0)

        chain_data[strike].update({
            f"{opt_type}_bid_usd": bid_usd,
            f"{opt_type}_ask_usd": ask_usd,
            f"{opt_type}_mark_usd": mark_usd,
            f"{opt_type}_bid_btc": bid_usd / underlying_price if underlying_price else 0,
            f"{opt_type}_ask_btc": ask_usd / underlying_price if underlying_price else 0,
            f"{opt_type}_mark_btc": mark_usd / underlying_price if underlying_price else 0,
            f"{opt_type}_iv": float(mark.get("markIV") or 0) * 100,
            f"{opt_type}_delta": float(mark.get("delta") or 0),
            f"{opt_type}_gamma": float(mark.get("gamma") or 0),
            f"{opt_type}_theta": float(mark.get("theta") or 0),
            f"{opt_type}_vega": float(mark.get("vega") or 0),
            f"{opt_type}_rho": 0,  # Binance doesn't provide Rho
            f"{opt_type}_oi": float(ticker.get("openInterest", 0) or 0),
            "underlying_price": underlying_price
        })

        if progress_callback:
            progress_callback(idx + 1, len(symbols), symbol)

    df = pd.DataFrame(list(chain_data.values()))
    return df.sort_values("strike").reset_index(drop=True) if not df.empty else df


# =============================================================================
# DUMP FUNCTIONS
# =============================================================================

def dump_options(exchange: str, days: int = 7):
    """Export option chains to CSV for given exchange"""
    is_deribit = exchange == "deribit"
    print(f"Dumping {exchange.upper()} options expiring within {days} days...")

    # Get expirations
    if is_deribit:
        expirations = deribit_get_expirations_within_days(days)
        format_exp = lambda x: x
    else:
        expirations = binance_get_expirations_within_days(days)
        format_exp = binance_format_expiry

    if not expirations:
        print("No expirations found within the specified timeframe.")
        return

    print(f"Found {len(expirations)} expiration(s): {', '.join(format_exp(e) for e in expirations)}")

    all_data = []

    for expiry in expirations:
        print(f"\nFetching {format_exp(expiry)}...", end=" ", flush=True)

        if is_deribit:
            df = deribit_get_options_chain(expiry, use_parallel=True)
        else:
            df = binance_get_options_chain(expiry)

        if not df.empty:
            df["exchange"] = exchange
            all_data.append(df)
            print(f"Got {len(df)} strikes")
        else:
            print("No data")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        # Reorder columns
        cols = [c for c in CSV_COLUMNS if c in combined.columns]
        combined = combined[cols]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exchange}_options_{days}d_{timestamp}.csv"
        combined.to_csv(filename, index=False)
        print(f"\nExported {len(combined)} rows to {filename}")
    else:
        print("No data to export.")


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_options_chain(df: pd.DataFrame, btc_price: float, exchange: str = "deribit"):
    """Display options chain in a formatted table"""
    is_deribit = exchange == "deribit"
    bid_col = "call_bid_btc" if is_deribit else "call_bid_usd"

    print("\n" + "=" * 130)
    print(f"{'CALLS':^60} | {'STRIKE':^8} | {'PUTS':^60}")
    print("=" * 130)
    print(f"{'IV%':>6} {'Delta':>7} {'Gamma':>9} {'Theta':>9} {'Vega':>8} {'Bid':>10} {'Ask':>10} | {'':^8} | "
          f"{'Bid':>10} {'Ask':>10} {'Vega':>8} {'Theta':>9} {'Gamma':>9} {'Delta':>7} {'IV%':>6}")
    print("-" * 130)

    for _, row in df.iterrows():
        strike = row["strike"]
        itm_c, itm_p = ">" if strike < btc_price else " ", "<" if strike > btc_price else " "
        atm = "*" if abs(strike - btc_price) / btc_price < 0.02 else " "

        def get(col, default=0):
            return row.get(col, default) or default

        c_iv, c_d, c_g, c_t, c_v = get("call_iv"), get("call_delta"), get("call_gamma"), get("call_theta"), get("call_vega")
        p_iv, p_d, p_g, p_t, p_v = get("put_iv"), get("put_delta"), get("put_gamma"), get("put_theta"), get("put_vega")

        if is_deribit:
            c_b, c_a = get("call_bid_btc"), get("call_ask_btc")
            p_b, p_a = get("put_bid_btc"), get("put_ask_btc")
            fmt = f"{c_iv:>6.1f} {c_d:>7.4f} {c_g:>9.7f} {c_t:>9.2f} {c_v:>8.2f} {c_b:>10.6f} {c_a:>10.6f} |{itm_c}{atm}{strike:>6}{atm}{itm_p}| {p_b:>10.6f} {p_a:>10.6f} {p_v:>8.2f} {p_t:>9.2f} {p_g:>9.7f} {p_d:>7.4f} {p_iv:>6.1f}"
        else:
            c_b, c_a = get("call_bid_usd"), get("call_ask_usd")
            p_b, p_a = get("put_bid_usd"), get("put_ask_usd")
            fmt = f"{c_iv:>6.1f} {c_d:>7.4f} {c_g:>9.7f} {c_t:>9.2f} {c_v:>8.2f} {c_b:>10.2f} {c_a:>10.2f} |{itm_c}{atm}{strike:>6}{atm}{itm_p}| {p_b:>10.2f} {p_a:>10.2f} {p_v:>8.2f} {p_t:>9.2f} {p_g:>9.7f} {p_d:>7.4f} {p_iv:>6.1f}"
        print(fmt)

    print("=" * 130)
    print(f"Current BTC Price: ${btc_price:,.2f} | Price Unit: {'BTC' if is_deribit else 'USD'}")
    print("Legend: > = ITM Call, < = ITM Put, * = Near ATM")


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def interactive_menu():
    """Interactive menu for the options viewer"""
    exchange = "deribit"

    while True:
        print(f"\n{'=' * 55}\n  BTC Options Viewer (No API Key Required)\n{'=' * 55}")
        print(f"  Current Exchange: {exchange.upper()}")

        try:
            btc_price = deribit_get_index_price() if exchange == "deribit" else binance_get_index_price()
            print(f"  Current BTC Price: ${btc_price:,.2f}")
        except Exception as e:
            print(f"  Error fetching BTC price: {e}")
            btc_price = 0

        print("\n  1. Switch exchange (Deribit/Binance)")
        print("  2. View available expiration dates")
        print("  3. View options chain (with Greeks)")
        print("  4. Export options chain to CSV")
        print("  5. Exit")

        choice = input("\n  Enter choice (1-5): ").strip()

        if choice == "1":
            print("\n  1. Deribit (prices in BTC, more expirations)")
            print("  2. Binance (prices in USD, faster API)")
            ex = input("  Enter choice (1-2): ").strip()
            exchange = "deribit" if ex == "1" else "binance" if ex == "2" else exchange
            print(f"  Switched to {exchange.upper()}")

        elif choice == "2":
            try:
                if exchange == "deribit":
                    exps = deribit_get_expiration_dates()
                else:
                    exps = [f"{binance_format_expiry(e)} ({e})" for e in binance_get_expiration_dates()]
                print(f"\n  Available Expiration Dates ({exchange.upper()}):")
                for i, exp in enumerate(exps, 1):
                    print(f"    {i}. {exp}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "3":
            try:
                exps = deribit_get_expiration_dates() if exchange == "deribit" else binance_get_expiration_dates()
                print(f"\n  Available Expiration Dates ({exchange.upper()}):")
                for i, exp in enumerate(exps, 1):
                    label = exp if exchange == "deribit" else f"{binance_format_expiry(exp)} ({exp})"
                    print(f"    {i}. {label}")

                exp_choice = input("\n  Enter expiry number or date: ").strip()
                expiry = exps[int(exp_choice) - 1] if exp_choice.isdigit() and 0 < int(exp_choice) <= len(exps) else exp_choice.upper()

                print(f"\n  Fetching options chain for {expiry}...")

                def progress(cur, tot, name):
                    print(f"  [{cur}/{tot}] {name}", end="\r")

                df = deribit_get_options_chain(expiry, progress, False) if exchange == "deribit" else binance_get_options_chain(expiry, progress)
                print()

                if not df.empty:
                    display_options_chain(df, btc_price, exchange)
                else:
                    print("  No options found for this expiry")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "4":
            try:
                exps = deribit_get_expiration_dates() if exchange == "deribit" else binance_get_expiration_dates()
                print(f"\n  Available Expiration Dates ({exchange.upper()}):")
                for i, exp in enumerate(exps, 1):
                    label = exp if exchange == "deribit" else f"{binance_format_expiry(exp)} ({exp})"
                    print(f"    {i}. {label}")

                exp_choice = input("\n  Enter expiry number or date: ").strip()
                expiry = exps[int(exp_choice) - 1] if exp_choice.isdigit() and 0 < int(exp_choice) <= len(exps) else exp_choice.upper()

                print(f"\n  Fetching options chain for {expiry}...")
                df = deribit_get_options_chain(expiry) if exchange == "deribit" else binance_get_options_chain(expiry)

                filename = f"btc_options_{exchange}_{expiry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print(f"\n  Exported to {filename}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "5":
            print("\n  Goodbye!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BTC Options Viewer - Fetch options data from Deribit and Binance (no API key required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python btc_options_viewer.py                  # Interactive menu
  python btc_options_viewer.py --dump-binance   # Export Binance options (7 days)
  python btc_options_viewer.py --dump-deribit   # Export Deribit options (7 days)
  python btc_options_viewer.py --dump-binance --days 14
        """
    )
    parser.add_argument("--dump-binance", action="store_true", help="Export Binance option chains to CSV")
    parser.add_argument("--dump-deribit", action="store_true", help="Export Deribit option chains to CSV")
    parser.add_argument("--days", type=int, default=7, help="Days from now to include expirations (default: 7)")

    args = parser.parse_args()

    if args.dump_binance:
        dump_options("binance", args.days)
    elif args.dump_deribit:
        dump_options("deribit", args.days)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lotto Scanner - Hybrid Deribit + Binance Options Intel.

Uses Deribit for GEX zone identification (better OI/liquidity).
Uses Binance for tradeable G/T ratios and WW bands.
Cross-references to find optimal lotto plays.

Usage:
    python lotto_scanner.py --days 7 --cost 5
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor

from btc_options_viewer import (
    deribit_get_index_price, deribit_get_expirations_within_days, deribit_get_options_chain,
    binance_get_index_price, binance_get_expirations_within_days, binance_get_options_chain,
    binance_format_expiry
)


def fetch_exchange_data(exchange, days):
    """Fetch option chain from a single exchange."""
    if exchange == "deribit":
        btc_price = deribit_get_index_price()
        expirations = deribit_get_expirations_within_days(days)
        get_chain = deribit_get_options_chain
        format_exp = lambda x: x
    else:
        btc_price = binance_get_index_price()
        expirations = binance_get_expirations_within_days(days)
        get_chain = binance_get_options_chain
        format_exp = binance_format_expiry

    if not expirations:
        return None, btc_price

    all_frames = []
    for exp in expirations:
        df = get_chain(exp)
        if not df.empty:
            df['expiry_date'] = format_exp(exp)
            df['exchange'] = exchange
            all_frames.append(df)

    if not all_frames:
        return None, btc_price

    return pd.concat(all_frames, ignore_index=True), btc_price


def fetch_both_exchanges(days):
    """Fetch from both Deribit and Binance in parallel."""
    print(f"Fetching options from both exchanges (expiring within {days} days)...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_deribit = executor.submit(fetch_exchange_data, "deribit", days)
        future_binance = executor.submit(fetch_exchange_data, "binance", days)
        
        deribit_df, deribit_price = future_deribit.result()
        binance_df, binance_price = future_binance.result()
    
    btc_price = binance_price or deribit_price  # Use Binance price as primary
    
    print(f"  Deribit: {len(deribit_df) if deribit_df is not None else 0} rows")
    print(f"  Binance: {len(binance_df) if binance_df is not None else 0} rows")
    
    return deribit_df, binance_df, btc_price


def calculate_gex(df, btc_price):
    """Calculate GEX for Deribit data (has OI)."""
    if df is None or df.empty:
        return df
    
    # Ensure columns exist
    for col in ['call_gamma', 'put_gamma', 'call_oi', 'put_oi']:
        if col not in df.columns:
            df[col] = 0.0
    
    df.fillna(0, inplace=True)
    
    spot_1pct = btc_price * 0.01
    df['call_gex'] = df['call_gamma'] * df['call_oi'] * btc_price * spot_1pct
    df['put_gex'] = df['put_gamma'] * df['put_oi'] * btc_price * spot_1pct
    df['total_gex'] = df['call_gex'] + df['put_gex']
    df['net_gex'] = df['call_gex'] - df['put_gex']  # Positive = call heavy
    df['dist_pct'] = (df['strike'] - btc_price) / btc_price * 100
    
    return df


def calculate_gamma_metrics(df, btc_price, cost_bps=5.0):
    """Calculate G/T ratio and WW band for Binance data."""
    if df is None or df.empty:
        return df
    
    cols = ['call_gamma', 'put_gamma', 'call_theta', 'put_theta', 
            'call_mark_usd', 'put_mark_usd', 'strike', 'call_iv', 'put_iv']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df.fillna(0, inplace=True)

    # Gamma Rent = Gamma / |Theta|
    df['call_gamma_rent'] = np.where(df['call_theta'] != 0, 
                                      df['call_gamma'] / df['call_theta'].abs(), 0)
    df['put_gamma_rent'] = np.where(df['put_theta'] != 0, 
                                     df['put_gamma'] / df['put_theta'].abs(), 0)

    # Whalley-Wilmott Bandwidth
    lam = cost_bps / 10000.0
    call_sigma = df['call_iv'].replace(0, np.nan) / 100.0
    put_sigma = df['put_iv'].replace(0, np.nan) / 100.0
    
    df['call_ww_band'] = ((1.5 * lam * btc_price * df['call_gamma']**2) / call_sigma) ** (1/3)
    df['put_ww_band'] = ((1.5 * lam * btc_price * df['put_gamma']**2) / put_sigma) ** (1/3)
    
    df['call_ww_band'] = df['call_ww_band'].fillna(0)
    df['put_ww_band'] = df['put_ww_band'].fillna(0)
    df['dist_pct'] = (df['strike'] - btc_price) / btc_price * 100

    return df


def find_gex_zones(deribit_df, btc_price, top_n=5):
    """Identify key GEX zones from Deribit."""
    if deribit_df is None or deribit_df.empty:
        return []
    
    # Aggregate GEX by strike (across expirations)
    gex_by_strike = deribit_df.groupby('strike').agg({
        'call_gex': 'sum',
        'put_gex': 'sum',
        'total_gex': 'sum',
        'net_gex': 'sum'
    }).reset_index()
    
    gex_by_strike['dist_pct'] = (gex_by_strike['strike'] - btc_price) / btc_price * 100
    
    # Filter to relevant range (+/- 15% from spot)
    gex_by_strike = gex_by_strike[abs(gex_by_strike['dist_pct']) <= 15]
    
    # Top Call GEX (gamma squeeze up)
    top_call_gex = gex_by_strike.nlargest(top_n, 'call_gex')
    
    # Top Put GEX (gamma squeeze down)
    top_put_gex = gex_by_strike.nlargest(top_n, 'put_gex')
    
    return top_call_gex, top_put_gex


def find_binance_plays_near_gex(binance_df, gex_zones, btc_price, tolerance_pct=2.0):
    """Find Binance options near Deribit GEX zones."""
    if binance_df is None or binance_df.empty:
        return pd.DataFrame()
    
    gex_strikes = set(gex_zones['strike'].tolist()) if not gex_zones.empty else set()
    
    # Find Binance strikes within tolerance of GEX zones
    def near_gex(strike):
        for gs in gex_strikes:
            if abs(strike - gs) / btc_price * 100 <= tolerance_pct:
                return True
        return False
    
    binance_df['near_gex'] = binance_df['strike'].apply(near_gex)
    
    return binance_df[binance_df['near_gex']]


def save_to_csv(results, scan_type="lotto_scan"):
    """Save scan results to CSV with timestamp."""
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lotto_scan_{scan_type}_{timestamp}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    if isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)
    else:
        # Handle dict/list of dicts
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
    
    print(f"\n💾 Saved results to: {filename}")
    return filepath


def save_to_json(results, scan_type="lotto_scan"):
    """Save scan results to JSON with timestamp."""
    if results is None:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lotto_scan_{scan_type}_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "scan_type": scan_type,
        "results": results if not isinstance(results, pd.DataFrame) else results.to_dict('records')
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Saved results to: {filename}")
    return filepath


def calculate_play_score(row, btc_price, is_call=True):
    """Calculate a simple score for ranking plays."""
    # Factors: distance to spot, premium, G/T ratio
    dist_weight = 30  # Prefer closer to spot
    premium_weight = 40  # Prefer lower premium
    gt_weight = 30  # Prefer higher G/T ratio
    
    # Normalize factors
    dist_score = max(0, 100 - abs(row['dist_pct']))
    
    # Premium score (lower = better) - cap at reasonable range
    premium = row.get('call_mark_usd' if is_call else 'put_mark_usd', 0)
    premium_score = max(0, 100 - min(premium, 100))
    
    # G/T ratio score (higher = better) - assume range 0-20
    gt = row.get('call_gamma_rent' if is_call else 'put_gamma_rent', 0)
    gt_score = min(100, gt * 5)
    
    total_score = (dist_score * dist_weight + 
                  premium_score * premium_weight + 
                  gt_score * gt_weight) / 100
    
    return total_score


def run_lotto_scan(days=7, cost_bps=5.0, export_csv=False, export_json=False):
    """Main scan combining both exchanges."""
    
    deribit_df, binance_df, btc_price = fetch_both_exchanges(days)
    
    print(f"\n{'='*60}")
    print(f"BTC Price: ${btc_price:,.2f} | Txn Cost: {cost_bps} bps")
    print(f"{'='*60}")
    
    # Calculate metrics
    deribit_df = calculate_gex(deribit_df, btc_price)
    binance_df = calculate_gamma_metrics(binance_df, btc_price, cost_bps)
    
    # Find GEX zones from Deribit
    top_call_gex, top_put_gex = find_gex_zones(deribit_df, btc_price)
    
    print("\n🎯 DERIBIT GEX ZONES (Gamma Walls - Market Wide)")
    print("-" * 60)
    
    print("\n📈 CALL GEX (Upside Squeeze Zones):")
    print(f"{'Strike':<10} | {'Dist%':>7} | {'Call GEX':>12} | {'Net GEX':>12}")
    print("-" * 50)
    for _, row in top_call_gex.iterrows():
        print(f"${row['strike']:<9,.0f} | {row['dist_pct']:>6.1f}% | ${row['call_gex']/1e6:>10.2f}M | ${row['net_gex']/1e6:>10.2f}M")
    
    print("\n📉 PUT GEX (Downside Squeeze Zones):")
    print(f"{'Strike':<10} | {'Dist%':>7} | {'Put GEX':>12} | {'Net GEX':>12}")
    print("-" * 50)
    for _, row in top_put_gex.iterrows():
        print(f"${row['strike']:<9,.0f} | {row['dist_pct']:>6.1f}% | ${row['put_gex']/1e6:>10.2f}M | ${row['net_gex']/1e6:>10.2f}M")
    
    # Binance tradeable plays
    print("\n\n🎰 BINANCE TRADEABLE PLAYS")
    print("-" * 60)
    
    if binance_df is not None and not binance_df.empty:
        # Filter liquid options (has bid)
        liquid_calls = binance_df[binance_df['call_bid_usd'] > 0].copy()
        liquid_puts = binance_df[binance_df['put_bid_usd'] > 0].copy()
        
        print("\n📈 BEST CALL PLAYS (by G/T Ratio):")
        print(f"{'Strike':<8} | {'Expiry':<9} | {'Dist%':>6} | {'Premium':>8} | {'G/T':>8} | {'WW Band':>8} | {'IV':>5}")
        print("-" * 70)
        
        top_calls = liquid_calls.nlargest(8, 'call_gamma_rent')
        for _, row in top_calls.iterrows():
            print(f"{row['strike']:<8.0f} | {row['expiry_date']:<9} | {row['dist_pct']:>5.1f}% | "
                  f"${row['call_mark_usd']:>7.1f} | {row['call_gamma_rent']:>8.4f} | "
                  f"+/-{row['call_ww_band']:>5.3f} | {row['call_iv']:>4.0f}%")
        
        print("\n📉 BEST PUT PLAYS (by G/T Ratio):")
        print(f"{'Strike':<8} | {'Expiry':<9} | {'Dist%':>6} | {'Premium':>8} | {'G/T':>8} | {'WW Band':>8} | {'IV':>5}")
        print("-" * 70)
        
        top_puts = liquid_puts.nlargest(8, 'put_gamma_rent')
        for _, row in top_puts.iterrows():
            print(f"{row['strike']:<8.0f} | {row['expiry_date']:<9} | {row['dist_pct']:>5.1f}% | "
                  f"${row['put_mark_usd']:>7.1f} | {row['put_gamma_rent']:>8.4f} | "
                  f"+/-{row['put_ww_band']:>5.3f} | {row['put_iv']:>4.0f}%")
    
    # Cross-reference: Binance plays near GEX zones
    all_gex_zones = pd.concat([top_call_gex, top_put_gex]).drop_duplicates(subset=['strike'])
    near_gex_plays = find_binance_plays_near_gex(binance_df, all_gex_zones, btc_price)
    
    if not near_gex_plays.empty:
        print("\n\n⚡ BINANCE PLAYS NEAR GEX ZONES (High Squeeze Potential)")
        print("-" * 60)
        print(f"{'Strike':<8} | {'Expiry':<9} | {'Dist%':>6} | {'Call $':>7} | {'Put $':>7} | {'Type':<6}")
        print("-" * 60)
        
        for _, row in near_gex_plays.drop_duplicates(subset=['strike', 'expiry_date']).head(10).iterrows():
            opt_type = "CALL" if row['dist_pct'] > 0 else "PUT"
            call_p = row.get('call_mark_usd', 0)
            put_p = row.get('put_mark_usd', 0)
            print(f"{row['strike']:<8.0f} | {row['expiry_date']:<9} | {row['dist_pct']:>5.1f}% | "
                  f"${call_p:>6.1f} | ${put_p:>6.1f} | {opt_type}")
    
    # Best Play Summary
    print("\n\n🎯 TOP RECOMMENDATION")
    print("=" * 60)
    
    all_plays = []
    
    # Get scored plays
    if binance_df is not None and not binance_df.empty:
        liquid_calls = binance_df[binance_df['call_bid_usd'] > 0].copy()
        liquid_puts = binance_df[binance_df['put_bid_usd'] > 0].copy()
        
        # Score calls
        liquid_calls['score'] = liquid_calls.apply(lambda r: calculate_play_score(r, btc_price, is_call=True), axis=1)
        liquid_puts['score'] = liquid_puts.apply(lambda r: calculate_play_score(r, btc_price, is_call=False), axis=1)
        
        all_plays.append(liquid_calls.nlargest(3, 'score'))
        all_plays.append(liquid_puts.nlargest(3, 'score'))
        
        # Add GEX zone plays with bonus
        if not near_gex_plays.empty:
            near_gex_plays['score'] = near_gex_plays.apply(lambda r: calculate_play_score(r, btc_price, r['dist_pct'] > 0), axis=1)
            all_plays.append(near_gex_plays.nlargest(3, 'score'))
    
    if all_plays:
        combined = pd.concat(all_plays, ignore_index=True)
        best_play = combined.nlargest(1, 'score').iloc[0]
        
        play_type = "CALL" if best_play['dist_pct'] > 0 else "PUT"
        premium = best_play.get('call_mark_usd' if best_play['dist_pct'] > 0 else 'put_mark_usd', 0)
        near_gex = "✓ NEAR GEX ZONE" if best_play.get('near_gex', False) else ""
        
        print(f"🥇 {play_type} ${best_play['strike']:,.0f} {best_play['expiry_date']}")
        print(f"   Premium: ${premium:.2f} | Distance: {best_play['dist_pct']:+.1f}%")
        print(f"   IV: {best_play.get('call_iv' if best_play['dist_pct'] > 0 else 'put_iv', 0):.0f}% | G/T: {best_play.get('call_gamma_rent' if best_play['dist_pct'] > 0 else 'put_gamma_rent', 0):.2f}")
        print(f"   Score: {best_play['score']:.1f} | {near_gex}")
        print("=" * 60)
    else:
        print("   No liquid plays found at this time.")
    
    # Export if requested
    if export_csv or export_json:
        results_to_export = combined if all_plays else pd.DataFrame()
        if export_csv:
            save_to_csv(results_to_export, "best_plays")
        if export_json:
            save_to_json(results_to_export, "best_plays")
    
    print("\n" + "=" * 60)
    print("Legend:")
    print("  GEX = Gamma Exposure. High GEX = potential squeeze zone")
    print("  G/T = Gamma/Theta ratio. Higher = more wiggle per decay")
    print("  WW Band = Hedge only when delta moves by this much")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Lotto Scanner - Hybrid Deribit + Binance Intel")
    parser.add_argument("--days", type=int, default=7, help="Days until expiry")
    parser.add_argument("--cost", type=float, default=5.0, help="Transaction cost (bps)")
    parser.add_argument("--csv", action="store_true", help="Export results to CSV")
    parser.add_argument("--json", action="store_true", help="Export results to JSON")
    args = parser.parse_args()
    
    run_lotto_scan(args.days, args.cost, export_csv=args.csv, export_json=args.json)


if __name__ == "__main__":
    main()

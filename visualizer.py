#!/usr/bin/env python3
"""Pilk Options Chain Visualizer - Improved Styling

Provides CLI tooling for building heat maps of Deribit open interest, overlaying GEX zones,
and plotting IV smiles for BTC options. Enhanced with professional styling.
"""

from __future__ import annotations

import ccxt
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from rich.console import Console
from rich.table import Table

# Set professional styling
sns.set_style("darkgrid")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['grid.alpha'] = 0.15
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

from btc_options_viewer import deribit_get_index_price, deribit_get_options_chain
from lotto_scanner import calculate_gex, find_gex_zones

def fetch_live_spot_price() -> float:
    """Get current BTC spot price using CCXT (Binance exchange)."""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        return ticker['last']
    except Exception as e:
        raise RuntimeError(f"Failed to fetch live spot price from CCXT: {e}")


def _normalize_format_list(raw: List[str], allowed: List[str], option_name: str) -> List[str]:
    normalized: List[str] = []
    for fmt in raw:
        candidate = fmt.strip().lower()
        if candidate not in allowed:
            raise typer.BadParameter(
                f"Unsupported value for {option_name}: '{fmt}'. Allowed: {', '.join(allowed)}"
            )
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


class VisualizerError(Exception):
    """Custom exception for the visualizer."""


console = Console()
app = typer.Typer(help="Pilk options chain visualizer CLI")


def fetch_deribit_data(expiry: str) -> tuple[pd.DataFrame, float]:
    """Fetch Deribit chain data and the current BTC spot price."""
    console.print(f"[bold]Fetching Deribit BTC chain for expiry [cyan]{expiry}[/cyan]...[/bold]")
    try:
        df = deribit_get_options_chain(expiry, use_parallel=True)
    except Exception as exc:  # pragma: no cover - API dependent
        raise VisualizerError(f"Unable to fetch Deribit data: {exc}") from exc

    if df.empty:
        raise VisualizerError(f"No options were returned for expiry {expiry}.")

    df = df.sort_values("strike").reset_index(drop=True)
    df["expiry"] = expiry

    btc_price = fetch_live_spot_price()
    return df, btc_price


def _display_gex_table(call_zones: pd.DataFrame, put_zones: pd.DataFrame) -> None:
    """Print GEX summary for the provided zones."""
    table = Table(title="GEX Zones", show_lines=True)
    table.add_column("Type", style="bold cyan")
    table.add_column("Strike", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Gamma Exposure", justify="right")
    table.add_column("Net GEX", justify="right")

    def add_rows(df: pd.DataFrame, label: str) -> None:
        if df.empty:
            table.add_row(label, "—", "—", "—", "—")
            return
        for _, row in df.head(5).iterrows():
            table.add_row(
                label,
                f"${row['strike']:,}",
                f"{row['dist_pct']:.1f}%",
                f"${row[f'{label.lower()}_gex'] / 1e6:.2f}M" if f"{label.lower()}_gex" in row else "N/A",
                f"${row['net_gex'] / 1e6:.2f}M",
            )

    add_rows(call_zones, "Call")
    add_rows(put_zones, "Put")
    console.print(table)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor='white', edgecolor='none')
    console.print(f"[green]Saved:[/green] {path}")
    return path


def build_heatmap(
    df: pd.DataFrame,
    call_zones: pd.DataFrame,
    put_zones: pd.DataFrame,
    btc_price: float,
    expiry: str,
) -> plt.Figure:
    """Build the open interest heat map with GEX overlays - improved styling."""
    strikes = df["strike"].astype(float).to_numpy()
    call_oi = df.get("call_oi", pd.Series(0, index=df.index)).fillna(0).to_numpy()
    put_oi = df.get("put_oi", pd.Series(0, index=df.index)).fillna(0).to_numpy()

    matrix = np.vstack([call_oi, put_oi])

    # Professional color palette - using 'YlOrRd' for warm, professional heatmap
    fig, ax = plt.subplots(figsize=(15, 7))

    heatmap_kwargs = {"cmap": "YlOrRd", "aspect": "auto", "origin": "lower"}

    positives = matrix[matrix > 0]
    if positives.size > 0:
        heatmap_kwargs["norm"] = LogNorm(vmin=max(positives.min(), 1e-2), vmax=positives.max())
        cbar_label = "Open Interest (log scale)"
    else:
        cbar_label = "Open Interest"

    im = ax.imshow(matrix, **heatmap_kwargs)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, aspect=30)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    x_positions = np.arange(len(strikes))
    step = max(1, len(strikes) // 12)
    tick_indices = x_positions[::step]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{int(strikes[i]):,}" for i in tick_indices], rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Call OI", "Put OI"], fontsize=11)

    # Improved title with better formatting
    title_text = f"Open Interest Heatmap — BTC {expiry} — Spot ${btc_price:,.0f}"
    ax.set_title(title_text, pad=15, fontweight='600', loc='left')
    ax.set_xlabel("Strike Price", fontsize=11, fontweight='500')
    ax.set_xlim(-0.5, len(strikes) - 0.5)
    ax.set_ylim(-0.5, 1.5)

    # Professional grid styling
    ax.grid(False)  # Turn off default grid for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    strike_to_idx = {strike: idx for idx, strike in enumerate(strikes)}

    def draw_zones(zones: pd.DataFrame, label: str, color: str) -> None:
        for idx, row in zones.head(4).iterrows():
            idx_pos = strike_to_idx.get(row["strike"])
            if idx_pos is None:
                continue
            # More subtle, professional dashed lines
            ax.axvline(idx_pos, color=color, linestyle='--', linewidth=2, alpha=0.7)
            ax.text(
                idx_pos,
                1.42,
                f"{label} GEX",
                rotation=90,
                color=color,
                fontsize=9,
                fontweight='600',
                va='bottom',
                ha='center',
                alpha=0.85,
            )

    # Professional color scheme for zones
    draw_zones(call_zones, "Call", "#2ECC71")  # Emerald green
    draw_zones(put_zones, "Put", "#E74C3C")  # Professional red

    # Enhanced legend
    legend_entries = []
    if not call_zones.empty:
        legend_entries.append(Line2D([0], [0], color="#2ECC71", linestyle='--', linewidth=2, label="Call GEX Zone"))
    if not put_zones.empty:
        legend_entries.append(Line2D([0], [0], color="#E74C3C", linestyle='--', linewidth=2, label="Put GEX Zone"))
    if legend_entries:
        ax.legend(
            handles=legend_entries,
            loc='upper left',
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            shadow=True,
            borderpad=0.5
        )

    # IV scatter overlay with professional styling
    avg_iv = np.nan_to_num((df.get("call_iv", 0) + df.get("put_iv", 0)) / 2)
    iv_axis = ax.twinx()
    scatter = iv_axis.scatter(
        x_positions,
        avg_iv,
        c=avg_iv,
        cmap="plasma",  # Professional alternative to viridis
        edgecolors='white',
        linewidths=0.8,
        s=70,
        alpha=0.85,
        zorder=4,
    )
    iv_axis.set_ylabel("Average IV (%)", fontsize=11, fontweight='500')
    iv_axis.set_ylim(0, max(avg_iv.max() * 1.2, 10))
    iv_axis.tick_params(axis='y', labelsize=9)
    iv_axis.grid(False)

    # Hide top and right spines for twinx
    iv_axis.spines['top'].set_visible(False)
    iv_axis.spines['right'].set_visible(False)

    iv_cbar = fig.colorbar(scatter, ax=ax, pad=0.015, aspect=30)
    iv_cbar.set_label("Avg IV (%)", fontsize=10)
    iv_cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


def build_iv_smile(df: pd.DataFrame, btc_price: float, expiry: str) -> plt.Figure:
    """Plot IV smile with calls vs puts - improved professional styling."""
    strikes = df["strike"].astype(float)
    call_iv = df.get("call_iv", 0).fillna(0)
    put_iv = df.get("put_iv", 0).fillna(0)
    dist_pct = ((strikes - btc_price) / btc_price) * 100

    # Professional color scheme using seaborn's palette
    call_color = "#3498db"  # Professional blue
    put_color = "#e74c3c"   # Professional red
    spot_color = "#7f8c8d"  # Professional gray

    norm = Normalize(vmin=-30, vmax=30)
    cmap = "coolwarm"

    fig, ax = plt.subplots(figsize=(14, 6))

    # Smoother, more professional lines
    ax.plot(strikes, call_iv, label="Call IV", color=call_color, linewidth=2.2, marker='o', markersize=4, alpha=0.9)
    ax.plot(strikes, put_iv, label="Put IV", color=put_color, linewidth=2.2, marker='s', markersize=4, alpha=0.9)

    # Professional scatter with better styling
    ax.scatter(
        strikes,
        call_iv,
        c=dist_pct,
        cmap=cmap,
        norm=norm,
        edgecolor='white',
        linewidth=0.8,
        s=50,
        alpha=0.7,
        zorder=3,
    )
    ax.scatter(
        strikes,
        put_iv,
        c=dist_pct,
        cmap=cmap,
        norm=norm,
        marker='s',
        edgecolor='white',
        linewidth=0.8,
        s=50,
        alpha=0.7,
        zorder=3,
    )

    # Enhanced spot line
    ax.axvline(btc_price, linestyle='--', color=spot_color, linewidth=1.8, alpha=0.7, label='Spot Price')

    # Improved title and labels
    title_text = f"IV Smile — BTC {expiry} — Spot ${btc_price:,.0f}"
    ax.set_title(title_text, pad=15, fontweight='600', loc='left')
    ax.set_xlabel("Strike Price ($)", fontsize=11, fontweight='500')
    ax.set_ylabel("Implied Volatility (%)", fontsize=11, fontweight='500')

    # Professional grid
    ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)

    # Enhanced legend
    ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        borderpad=0.5,
        ncol=1
    )

    # Professional colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("Distance from Spot (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Set spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    fig.tight_layout()
    return fig


@app.command()
def visualize(
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Underlying symbol (BTC only for now)."),
    expiry: str = typer.Option(..., "--expiry", "-e", help="Deribit expiry (e.g., 07FEB26)."),
    show: bool = typer.Option(
        True,
        "--show/--no-show",
        help="Show the Matplotlib windows after building the visualizations.",
    ),
    save_dir: Optional[Path] = typer.Option(
        None,
        "--save-dir",
        "-d",
        help="Optional directory to save PNG previews for the heat map + IV smile (default: none).",
    ),
) -> None:
    symbol = symbol.upper()
    if symbol != "BTC":
        raise typer.BadParameter("Only BTC is currently supported.")

    try:
        df, btc_price = fetch_deribit_data(expiry)
    except VisualizerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    df = calculate_gex(df, btc_price)
    call_zones, put_zones = find_gex_zones(df, btc_price)

    _display_gex_table(call_zones, put_zones)

    heatmap_fig = build_heatmap(df, call_zones, put_zones, btc_price, expiry)
    iv_fig = build_iv_smile(df, btc_price, expiry)

    saved_paths: List[Path] = []
    if save_dir:
        save_dir = save_dir.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = _save_figure(heatmap_fig, save_dir / f"{symbol}_{expiry}_heatmap.png")
        iv_path = _save_figure(iv_fig, save_dir / f"{symbol}_{expiry}_iv_smile.png")
        saved_paths.extend([heatmap_path, iv_path])

    if show:
        plt.show()

    plt.close("all")

    if saved_paths:
        console.print(
            "[bold green]Saved visual previews:[/bold green] \n" + "\n".join(f" - {path}" for path in saved_paths)
        )


@app.command()
def export(
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Underlying symbol (BTC only)."),
    expiry: str = typer.Option(..., "--expiry", "-e", help="Deribit expiry (e.g., 07FEB26)."),
    format: List[str] = typer.Option(
        ["png"],
        "--format",
        "-f",
        help="Image formats for heat map + IV smile (png/svg).",
    ),
    data_format: List[str] = typer.Option(
        ["csv"],
        "--data-format",
        help="Raw data export formats (csv/json).",
    ),
    output_dir: Path = typer.Option(
        Path("visualizer_outputs"),
        "--output-dir",
        "-o",
        help="Directory where exported files will be written.",
    ),
) -> None:
    symbol = symbol.upper()
    if symbol != "BTC":
        raise typer.BadParameter("Only BTC is currently supported.")

    try:
        df, btc_price = fetch_deribit_data(expiry)
    except VisualizerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    df = calculate_gex(df, btc_price)
    call_zones, put_zones = find_gex_zones(df, btc_price)

    image_formats = _normalize_format_list(format, ["png", "svg"], "format")
    data_formats = _normalize_format_list(data_format, ["csv", "json"], "data-format")

    heatmap_fig = build_heatmap(df, call_zones, put_zones, btc_price, expiry)
    iv_fig = build_iv_smile(df, btc_price, expiry)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{symbol}_{expiry}_{timestamp}"

    saved_images: List[Path] = []
    for fmt in image_formats:
        heatmap_path = output_dir / f"{base}_heatmap.{fmt}"
        iv_path = output_dir / f"{base}_iv_smile.{fmt}"
        saved_images.append(_save_figure(heatmap_fig, heatmap_path))
        saved_images.append(_save_figure(iv_fig, iv_path))

    raw_paths: List[Path] = []
    raw_df = df.copy()
    raw_df["avg_iv"] = np.nan_to_num((raw_df.get("call_iv", 0) + raw_df.get("put_iv", 0)) / 2)

    if "csv" in data_formats:
        csv_path = output_dir / f"{base}_data.csv"
        raw_df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved:[/green] {csv_path}")
        raw_paths.append(csv_path)
    if "json" in data_formats:
        json_path = output_dir / f"{base}_data.json"
        with open(json_path, "w") as fp:
            json.dump(raw_df.to_dict("records"), fp, indent=2)
        console.print(f"[green]Saved:[/green] {json_path}")
        raw_paths.append(json_path)

    plt.close("all")

    console.print("[bold]Export complete![/bold]")
    console.print("[bold cyan]Images:[/bold cyan]", ", ".join(str(p) for p in saved_images))
    if raw_paths:
        console.print("[bold cyan]Raw data:[/bold cyan]", ", ".join(str(p) for p in raw_paths))


if __name__ == "__main__":
    app()

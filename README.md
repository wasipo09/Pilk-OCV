# 📊 Pilk-OCV — Options Chain Visualizer

A powerful CLI tool for visualizing cryptocurrency options chains with heat maps, GEX zones overlay, and IV (implied volatility) smile curves. Built for BTC/ETH options traders who need to quickly analyze market structure and identify trading opportunities.

---

## ✨ Features

### 🎨 Options Chain Heat Map
- **Open Interest Visualization:** Color-coded heat map of calls vs puts by strike and expiry
- **Volume Overlay:** Scatter points showing volume distribution
- **GEX Zones:** Highlight gamma exposure walls (potential squeeze zones)
- **IV Scatter:** Average implied volatility overlay with color bar

### 📈 IV Smile Plotter
- **Volatility Curve:** Plot IV across moneyness for calls and puts separately
- **Moneyness Color Coding:** Visualize OTM/ATM/ITM options
- **Spot Annotation:** Current price marked on the curve
- **Multiple Expiries:** Compare IV smiles across different expiry dates

### 💾 Export Options
- **Image Formats:** PNG, SVG for reports and presentations
- **Data Formats:** CSV, JSON for further analysis
- **Bulk Export:** Generate all visualizations with one command
- **Custom Output:** Specify output directory for organized exports

### 🔧 Developer-Friendly
- **CLI Interface:** Typer-based command-line tool (consistent with Pilk projects)
- **Rich Output:** Beautiful terminal output with progress indicators
- **Modular Design:** Reuses Deribit fetchers and GEX helpers from existing tools
- **Easy Integration:** Can be imported as a module for custom workflows

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Deribit API access (no key required for public data)

### Setup

```bash
# Clone the repository
git clone https://github.com/wasipo09/Pilk-OCV.git
cd Pilk-OCV

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `numpy` — Numerical operations
- `rich` — Terminal output formatting
- `typer` — CLI framework
- `matplotlib` — Plotting and visualization

---

## 📖 Usage

### Visualize Options Chain

```bash
# Basic visualization (BTC, 7FEB26 expiry)
python3 visualizer.py visualize --symbol BTC --expiry 7FEB26

# Show plots with interactive window
python3 visualizer.py visualize --symbol ETH --expiry 7FEB26 --show

# Don't display plots (save only)
python3 visualizer.py visualize --symbol BTC --expiry 7FEB26 --no-show
```

**What you get:**
- Heat map of open interest (calls vs puts by strike)
- GEX zones highlighted with annotations
- IV scatter overlay showing volatility distribution
- Console output with GEX summary

### IV Smile Analysis

```bash
# Plot IV smile (calls vs puts)
python3 visualizer.py visualize --symbol BTC --expiry 7FEB26

# Compare across expiries
python3 visualizer.py visualize --symbol BTC --expiry 7FEB26 --expiry 14FEB26

# Custom styling
python3 visualizer.py visualize --symbol BTC --expiry 7FEB26 --style dark
```

**What you get:**
- IV smile curve for calls (blue)
- IV smile curve for puts (red)
- Moneyness color coding (OTM, ATM, ITM)
- Spot price annotation

### Export Visualizations

```bash
# Export to PNG
python3 visualizer.py export --symbol BTC --expiry 7FEB26 --format png

# Export to SVG (vector graphics)
python3 visualizer.py export --symbol BTC --expiry 7FEB26 --format svg

# Export multiple formats
python3 visualizer.py export --symbol BTC --expiry 7FEB26 --format png --format svg

# Export data only
python3 visualizer.py export --symbol BTC --expiry 7FEB26 --data-format csv
python3 visualizer.py export --symbol BTC --expiry 7FEB26 --data-format json

# Custom output directory
python3 visualizer.py export --symbol BTC --expiry 7FEB26 --output-dir ./exports

# Export everything (images + data)
python3 visualizer.py export --symbol BTC --expiry 7FEB26 \
  --format png --format svg \
  --data-format csv --data-format json \
  --output-dir ./exports
```

**Output files:**
- `[SYMBOL]_[EXPIRY]_[TIMESTAMP]_heatmap.png/svg` — Options chain heat map
- `[SYMBOL]_[EXPIRY]_[TIMESTAMP]_iv_smile.png/svg` — IV smile plot
- `[SYMBOL]_[EXPIRY]_[TIMESTAMP]_data.csv` — Raw options data
- `[SYMBOL]_[EXPIRY]_[TIMESTAMP]_data.json` — Structured JSON data

---

## 📊 Examples

### Example 1: Quick BTC Options Analysis

```bash
# Visualize weekly options
python3 visualizer.py visualize --symbol BTC --expiry 7FEB26

# Console output:
# Fetching Deribit BTC chain for expiry 7FEB26...
# GEX Zones: 78K-80K (strong walls)
# Top Call GEX: $78,000 (OI: 3,124)
# Top Put GEX: $76,000 (OI: 2,548)
# Heat map saved to: BTC_7FEB26_20260204_120000_heatmap.png
# IV smile saved to: BTC_7FEB26_20260204_120000_iv_smile.png
```

### Example 2: Export for Reports

```bash
# Generate high-resolution visualizations
python3 visualizer.py export --symbol BTC --expiry 7FEB26 \
  --format png --data-format json \
  --output-dir ./reports

# Use PNGs in presentations, JSON for custom analysis
```

### Example 3: Batch Expiry Comparison

```bash
# Compare IV smiles across expiries
for expiry in 7FEB26 14FEB26 21FEB26; do
  python3 visualizer.py export --symbol BTC --expiry $expiry --format png
done

# Analyze volatility term structure and identify best expiry for trading
```

---

## 🏗️ Project Structure

```
Pilk-OCV/
├── visualizer.py       # Main CLI entry point
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── exports/          # Default output directory (created automatically)
```

---

## 🔬 Technical Details

### Data Sources
- **Deribit API** — Real-time options chain data (no API key required)
- **GEX Calculations** — Gamma exposure derived from option Greeks
- **IV Computation** — Implied volatility from option prices

### Visualization Methods
- **Heat Map:** `matplotlib.pyplot.imshow()` with color-coded OI intensity
- **Scatter Overlay:** `matplotlib.pyplot.scatter()` for volume and IV points
- **IV Smile:** Line plots with `matplotlib.pyplot.plot()` for calls/puts
- **Annotations:** Text and arrows for GEX zones and spot price

### Color Coding
- **Heat Map:** Yellow = High OI, Blue = Low OI
- **IV Points:** Red = High IV, Blue = Low IV
- **Moneyness:** Green = ITM, Yellow = ATM, Blue = OTM

---

## 🎯 Use Cases

### 📈 Traders
- Identify GEX zones for potential squeezes
- Spot unusual IV skews indicating market sentiment
- Compare expiries to find best value
- Visualize open interest concentration

### 📊 Analysts
- Generate reports with professional visualizations
- Export data for custom analysis
- Track IV changes over time
- Compare options across multiple symbols

### 🤖 Developers
- Import as module for custom workflows
- Reuse Deribit API fetchers
- Extend with custom visualizations
- Build on top of existing codebase

---

## 🔮 Future Enhancements

- [ ] Real-time streaming updates (WebSocket support)
- [ ] Multi-symbol comparison (BTC vs ETH side-by-side)
- [ ] Historical data visualization (IV surface over time)
- [ ] Custom timeframes (hourly, daily, weekly)
- [ ] Export to interactive HTML with Plotly
- [ ] Options strategy payoff diagrams

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 License

MIT License — feel free to use and modify for your needs.

---

## 👤 Author

Created for Pilk projects.

---

## 🙏 Acknowledgments

- **Deribit API** — Options data source
- **Matplotlib** — Visualization library
- **Rich** — Terminal output formatting
- **Typer** — CLI framework

---

**Happy Trading! 🎰💰**

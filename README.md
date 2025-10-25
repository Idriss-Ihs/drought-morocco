# 🌦️ Drought Monitoring — Morocco (SPI-Based)

**A data-driven system for monitoring drought conditions across Moroccan provinces using CHIRPS rainfall data and the Standardized Precipitation Index (SPI).**

---

## 🚀 Overview

This project builds a full end-to-end drought monitoring workflow — from satellite precipitation data to analytics and visualization.

**Key Features**

- Automated **CHIRPS** data download (1981–present)
- Country subset (Morocco + Western Sahara)
- **SPI computation** at 1, 3, 6, and 12-month scales (Gamma distribution fit)
- Provincial (ADM2) aggregation of SPI
- Yearly drought metrics: frequency, severity, persistence
- **Automated HTML report** generation
- **Interactive Streamlit dashboard** for exploration

---

## 🧩 Data Sources

- 🌍 **CHIRPS v2.0** — Climate Hazards Center InfraRed Precipitation with Station data
- 🗺️ **GeoBoundaries (ADM2)** — Official administrative boundaries (including Western Sahara)
- 🕒 **Period:** 1981–2025 (monthly)

---

## 🧠 Methodology

1. **Precipitation anomalies → SPI:**
   - Monthly precipitation aggregated to rolling sums (1–12 months)
   - Fitted to a **Gamma distribution** via Method of Moments
   - Zero-inflation correction for dry months
   - Cumulative probability transformed via Φ⁻¹ to yield SPI
2. **Aggregation:**
   - Mean SPI computed for each province per month
3. **Metrics:**
   - Drought/wetness frequency per class (mild, moderate, severe, extreme)
   - Maximum drought spell duration
   - National mean SPI trend slope (long-term signal)

---

## 📊 Outputs

| Type              | Description                    | Example                        |
| ----------------- | ------------------------------ | ------------------------------ |
| `data/processed/` | SPI & provincial drought stats | `provincial_drought_stats.csv` |
| `docs/figures/`   | Maps, trends, and leaderboards | `map_mean_spi_spi12_2025.png`  |
| `docs/report/`    | Auto-generated HTML summary    | `index.html`                   |
| `src/app/app.py`  | Streamlit dashboard            | Interactive analysis           |

---

## 🧾 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python -m src.data.download_chirps
python -m src.data.subset_morocco
python -m src.features.spi_compute
python -m src.features.aggregate_provinces
python -m src.features.compute_drought_stats

# 3. Build report
python -m src.report.build_report
# → open docs/report/index.html

# 4. Launch dashboard
streamlit run src/app/app.py
```

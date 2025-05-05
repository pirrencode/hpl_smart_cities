# Smart City GDP Model

Simulates how the introduction of a **passenger- and freight-Hyperloop corridor** through a smart city affects economic and transport metrics, including GDP shifts, commute accessibility, and node network changes.

---

## Overview
This model calculates:

- **GDP impact** (baseline, TRL-6, TRL-9) using a Poisson Pseudo-Maximum-Likelihood (PPML) accessibility model.
- **Commuting reach** by evaluating the number of 0.33 km² sectors accessible.
- **Network stats** including grid coverage and node density before and after Hyperloop introduction.

---

## Authors
- Program: **Aleksejs Vesjolijs**
- Model Contributors (PPML): **Aleksejs Vesjolijs**, **Yulia Stukalina**, **Olga Zervina**

---

## Input
**File:** `smc_input.csv`

Each row = one city. Required/optional columns:

| Column                     | Description                           | Default |
|---------------------------|---------------------------------------|---------|
| smc_name                  | City name                             | -       |
| smc_area                  | City area in km²                     | -       |
| smc_gdp_current           | GDP in billions USD                   | -       |
| current_commute_speed_kmh| Avg. commute speed                    | 30      |
| current_commute_time_min | One-way commute time (min)           | 60      |
| hyperloop_speed_trl6_kmh | HL TRL-6 speed                        | 450     |
| hyperloop_speed_trl9_kmh | HL TRL-9 speed                        | 1223    |
| hyperloop_route_length_km| HL route length (km)                  | 10      |
| current_exports           | Total exports (USD)                   | 0       |
| current_imports           | Total imports (USD)                   | 0       |
| current_freight_speed     | Freight transport speed (km/h)        | 90      |
| citizen_commute_time      | Round-trip commute time (min)        | 60      |
| smc_nodes                 | Nodes/km² in current network         | 35      |

---

## Output
**File:** `smart_city_metrics_output_ppml.csv`

Key output columns:

### Economic:
- `smc_gdp_trl6_final`, `smc_gdp_trl9_final`
- `smc_gdp_trl6_final_diff`, `smc_gdp_trl9_final_diff`

### Grid & Reach:
- `smc_grid`, `smc_grid_reach_*`, `hl_route_grid`

### Node Stats:
- `smc_nodes_with_hl`, `smc_nodes_increase_total`
- `smc_nodes_per_grid_current`, `smc_nodes_per_grid_with_hl`

---

## How It Works

1. **Load input** from `smc_input.csv`
2. **`calculate_smc_metrics(df)`**
    - Applies defaults
    - Calculates grid size, reach, GDP, trade, and node stats
3. **`make_ppml_frame(df)`**
    - Converts wide-format to long panel format
4. **`apply_ppml(df)`**
    - Runs or applies PPML GDP estimation
5. **Results merged**, adjusted for trade changes, and exported to CSV

---

## Run
```bash
python smart_city_gdp_model.py
```

This generates:

- `smart_city_metrics_output_ppml.csv`  
  (Tidy-wide dataset with all key outputs)
- Prints selected columns to stdout

---

## Dependencies

- `numpy`
- `pandas`
- `statsmodels`

Install with:
```bash
pip install numpy pandas statsmodels
```

---

## License
AGPL

---

## Acknowledgments


"""
smart_city_gdp_model.py
───────────────────────────────────────────────────────────────────────────────
Hyperloop × Smart‑City Integrated Model
───────────────────────────────────────────────────────────────────────────────
Simulates how introducing a passenger‑ and freight‑Hyperloop corridor through
a smart‑city affects

• GDP (baseline, TRL‑6, TRL‑9) via a Poisson Pseudo‑Maximum‑Likelihood (PPML)
  accessibility model plus a speed‑driven export boost,
• Reach: number of grid sectors (0.33 km² hexes) people can commute to,
• Transport‑network statistics: total grid size, sectors traversed by the
  Hyperloop route, and how many connection nodes per km² / per‑sector
  exist before and after Hyperloop.

───────────────────────────────────────────────────────────────────────────────
High‑level Flow
───────────────────────────────────────────────────────────────────────────────
1. **Read input CSV** (`smc_input.csv`) containing one row per city.
2. **calculate_smc_metrics**
       • fills missing defaults,
       • builds baseline grid, reach, GDP, trade, node statistics,
       • derives TRL‑6 / TRL‑9 values.
3. **make_ppml_frame** – converts wide → long panel (baseline, TRL‑6, TRL‑9).
4. **apply_ppml** – fits or applies a PPML gravity‑style model.
5. **Merge** PPML predictions back, add trade deltas → final GDP.
6. **Write** tidy wide CSV (`smart_city_metrics_output_ppml.csv`) &
   print selected columns.

───────────────────────────────────────────────────────────────────────────────
Expected Input Columns  († = optional / defaults provided)
───────────────────────────────────────────────────────────────────────────────
smc_name                 : city label  
smc_area                 : total area in km²  
smc_gdp_current          : current GDP **in billions USD**  
current_commute_speed_kmh†   (default 30)  
current_commute_time_min†    (default 60) → one‑way minutes  
hyperloop_speed_trl6_kmh†    (default 450)  
hyperloop_speed_trl9_kmh†    (default 1 223)  
hyperloop_route_length_km†   (default 10)  
current_exports†, current_imports†  (absolute USD, default 0)  
current_freight_speed†        (default 90 km h⁻¹)  
citizen_commute_time†         (default 60 min round‑trip)  
smc_nodes†                    (baseline connection nodes / km², default 35)

───────────────────────────────────────────────────────────────────────────────
Key Output Columns
───────────────────────────────────────────────────────────────────────────────
• Economic:
    smc_gdp_trl6_final, smc_gdp_trl9_final, …_diff
• Grid & reach:
    smc_grid, smc_grid_reach_current / _trl6 / _trl9,
    smc_grid_curr_diff_trl6 / _trl9, hl_route_grid
• Node statistics (purely descriptive):
    smc_nodes_with_hl            – nodes / km² after Hyperloop
    smc_nodes_per_grid_current   – baseline nodes per 0.33 km² sector
    smc_nodes_per_grid_with_hl   – same after Hyperloop
    smc_nodes_increase_total     – absolute number of new nodes city‑wide

───────────────────────────────────────────────────────────────────────────────
Running
───────────────────────────────────────────────────────────────────────────────
$ python smart_city_gdp_model.py

Produces:
    smart_city_metrics_output_ppml.csv    # tidy‑wide results
and prints the main columns to stdout.

Dependencies: numpy, pandas, statsmodels.
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# ---------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------
SECTOR_AREA_KM2                = 1/3
CURRENT_COMMUTE_SPEED_KMH_DEF  = 30
CURRENT_COMMUTE_TIME_MIN_DEF   = 60
HYPERLOOP_SPEED_TRL6_KMH_DEF   = 450
HYPERLOOP_SPEED_TRL9_KMH_DEF   = 1_223
HYPERLOOP_ROUTE_LENGTH_KM_DEF  = 10
SMC_NODES_DEF                  = 35

ESTIMATE_PPML = True
BETA_FIXED = {'const': -0.52, 'ln_dist': 0.71,
              'ln_time': 0.33, 'hl_dummy': 0.19}

# ---------------------------------------------------------------
# 2. SIMULATION
# ---------------------------------------------------------------
def calculate_smc_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    defaults = {
        'current_commute_speed_kmh': CURRENT_COMMUTE_SPEED_KMH_DEF,
        'current_commute_time_min' : CURRENT_COMMUTE_TIME_MIN_DEF,
        'hyperloop_speed_trl6_kmh' : HYPERLOOP_SPEED_TRL6_KMH_DEF,
        'hyperloop_speed_trl9_kmh' : HYPERLOOP_SPEED_TRL9_KMH_DEF,
        'hyperloop_route_length_km': HYPERLOOP_ROUTE_LENGTH_KM_DEF,
        'current_exports'          : 0.0,
        'current_imports'          : 0.0,
        'current_freight_speed'    : 90,
        'citizen_commute_time'     : 60,
        'smc_nodes'                : SMC_NODES_DEF
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df[['current_exports','current_imports']] = \
        df[['current_exports','current_imports']].clip(lower=0)

    results, rng = [], np.random.default_rng(42)
    sqrt_sector  = np.sqrt(SECTOR_AREA_KM2)

    for _, row in df.iterrows():
        name  = row['smc_name']
        area  = row['smc_area']
        gdp0  = row['smc_gdp_current'] * 1_000_000_000

        # ------------ grid size ------------------------------
        smc_grid = int(area / SECTOR_AREA_KM2)

        # =====================================================
        # === NODES BLOCK BEGIN ===============================
        # baseline nodes
        nodes_density_base = row['smc_nodes']
        nodes_total_base   = nodes_density_base * area

        # passenger HL
        hl_route_grid = int(row['hyperloop_route_length_km'] / sqrt_sector)
        nodes_passenger = hl_route_grid

        # freight HL 
        nodes_freight   = int(0.5 * hl_route_grid)

        nodes_added_total = nodes_passenger + nodes_freight
        nodes_density_hl  = (nodes_total_base + nodes_added_total) / area
        nodes_per_grid_base = nodes_density_base * SECTOR_AREA_KM2
        nodes_per_grid_hl   = nodes_density_hl  * SECTOR_AREA_KM2
        # === NODES BLOCK END =================================
        # =====================================================

        # ------------ baseline trade & speed -----------------
        x0, m0  = row['current_exports'], row['current_imports']
        fs0     = max(1, row['current_freight_speed'])
        net0    = x0 - m0

        c_spd   = row['current_commute_speed_kmh']
        c_time  = row['current_commute_time_min']
        s6, s9  = row['hyperloop_speed_trl6_kmh'], row['hyperloop_speed_trl9_kmh']
        hl_len  = row['hyperloop_route_length_km']
        cit_rt  = row['citizen_commute_time']

        dist0   = c_spd * c_time / 60
        reach0  = int(dist0 / sqrt_sector)

        dist6   = s6 * c_time / 60
        reach6  = int(dist6 / sqrt_sector)

        dist9   = s9 * c_time / 60
        reach9  = int(dist9 / sqrt_sector)

        diff6 = 100*(reach6-reach0)/reach0 if reach0 else 0
        diff9 = 100*(reach9-reach0)/reach0 if reach0 else 0

        time_factor = c_time / (24 * cit_rt)
        adj6 = 1 + (reach6/reach0 - 1)*time_factor
        adj9 = 1 + (reach9/reach0 - 1)*time_factor

        gdp_growth_6 = gdp0 * adj6
        gdp_growth_9 = gdp0 * adj9

        gdp_line = rng.uniform(2e6,6e6,size=hl_route_grid).sum()

        exp6, exp9 = x0*(s6/fs0), x0*(s9/fs0)
        imp6, imp9 = m0, m0
        net6, net9 = exp6-imp6, exp9-imp9

        gdp6_base = gdp_growth_6 + gdp_line
        gdp9_base = gdp_growth_9 + gdp_line

        results.append({
            'smc_name'                 : name,
            'smc_gdp_current'          : gdp0,

            # trade
            'current_exports'          : x0,
            'current_imports'          : m0,
            'current_net_trade'        : net0,
            'exports_trl6'             : exp6,
            'imports_trl6'             : imp6,
            'exports_trl9'             : exp9,
            'imports_trl9'             : imp9,
            'net_trade_trl6'           : net6,
            'net_trade_trl9'           : net9,

            # grid stats
            'smc_grid'                 : smc_grid,
            'smc_grid_reach_current'   : reach0,
            'smc_grid_reach_trl6'      : reach6,
            'smc_grid_curr_diff_trl6'  : diff6,
            'smc_grid_reach_trl9'      : reach9,
            'smc_grid_curr_diff_trl9'  : diff9,
            'hl_route_grid'            : hl_route_grid,

            # node stats
            'smc_nodes_with_hl'        : nodes_density_hl,
            'smc_nodes_per_grid_current': nodes_per_grid_base,
            'smc_nodes_per_grid_with_hl': nodes_per_grid_hl,
            'smc_nodes_increase_total' : nodes_added_total,

            # GDP
            'smc_gdp_trl6_base'        : gdp6_base,
            'smc_gdp_trl9_base'        : gdp9_base,
            'smc_gdp_trl6_final'       : gdp6_base,
            'smc_gdp_trl9_final'       : gdp9_base,

            # commute metrics
            'dist_curr_km'             : dist0,
            'dist_trl6_km'             : dist6,
            'dist_trl9_km'             : dist9,
            'time_red_trl6'            : c_time - c_time*reach0/reach6,
            'time_red_trl9'            : c_time - c_time*reach0/reach9,
        })

    return pd.DataFrame(results).round(2)

# ---------------------------------------------------------------
# 3. LONG PANEL FOR PPML
# ---------------------------------------------------------------
def make_ppml_frame(df_wide: pd.DataFrame) -> pd.DataFrame:
    sqrt_sec = np.sqrt(SECTOR_AREA_KM2)
    base = df_wide.assign(
        scenario='baseline', hl_dummy=0,
        distance_km=df_wide['smc_grid_reach_current']*sqrt_sec,
        time_red_min=0,
        gdp=df_wide['smc_gdp_current']
    )[['smc_name','scenario','distance_km','time_red_min','hl_dummy','gdp']]

    trl6 = df_wide.assign(scenario='TRL6', hl_dummy=1,
        distance_km=df_wide['smc_grid_reach_trl6']*sqrt_sec,
        time_red_min=df_wide['time_red_trl6'],
        gdp=df_wide['smc_gdp_trl6_final'])[base.columns]

    trl9 = df_wide.assign(scenario='TRL9', hl_dummy=1,
        distance_km=df_wide['smc_grid_reach_trl9']*sqrt_sec,
        time_red_min=df_wide['time_red_trl9'],
        gdp=df_wide['smc_gdp_trl9_final'])[base.columns]

    return pd.concat([base, trl6, trl9], ignore_index=True)

# ---------------------------------------------------------------
# 4. PPML
# ---------------------------------------------------------------
def apply_ppml(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    df['gdp_pos'] = df['gdp'].clip(lower=1)

    if ESTIMATE_PPML:
        mod = smf.glm("gdp_pos ~ np.log(distance_km) + "
                      "np.log(time_red_min+1) + hl_dummy",
                      data=df,
                      family=sm.families.Poisson(sm.families.links.log())
                     ).fit(cov_type='HC0')
        print(summary_col([mod], stars=True))
        df['gdp_ppml_pred'] = mod.predict(df)
    else:
        xb = (BETA_FIXED['const']
              + BETA_FIXED['ln_dist']*np.log(df['distance_km'])
              + BETA_FIXED['ln_time']*np.log(df['time_red_min']+1)
              + BETA_FIXED['hl_dummy']*df['hl_dummy'])
        df['gdp_ppml_pred'] = np.exp(xb)

    return df[['smc_name','scenario','gdp_ppml_pred']]

# ---------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    df_input = pd.read_csv("smc_input.csv")

    wide  = calculate_smc_metrics(df_input)
    long  = make_ppml_frame(wide)
    preds = apply_ppml(long)

    preds_wide = (preds.pivot(index='smc_name', columns='scenario',
                              values='gdp_ppml_pred')
                         .rename(columns={'TRL6':'gdp_ppml_trl6',
                                          'TRL9':'gdp_ppml_trl9'})
                         .reset_index())

    wide_ppml = (wide.merge(preds_wide, on='smc_name', how='left')
                 .assign(smc_gdp_trl6_final=lambda d:
                             d['gdp_ppml_trl6'] +
                             (d['net_trade_trl6'] - d['current_net_trade']),
                         smc_gdp_trl9_final=lambda d:
                             d['gdp_ppml_trl9'] +
                             (d['net_trade_trl9'] - d['current_net_trade']))
                 .drop(columns=['gdp_ppml_trl6','gdp_ppml_trl9'])
                 .assign(smc_gdp_trl6_final_diff=lambda d:
                             100*(d['smc_gdp_trl6_final']-d['smc_gdp_current'])
                             / d['smc_gdp_current'],
                         smc_gdp_trl9_final_diff=lambda d:
                             100*(d['smc_gdp_trl9_final']-d['smc_gdp_current'])
                             / d['smc_gdp_current'])
                 .round(2))

    cols = ['smc_name',
            'smc_gdp_current',
            'smc_gdp_trl6_final','smc_gdp_trl6_final_diff',
            'smc_gdp_trl9_final','smc_gdp_trl9_final_diff',
            'current_exports','current_imports','current_net_trade',
            'exports_trl6','imports_trl6','net_trade_trl6',
            'exports_trl9','imports_trl9','net_trade_trl9',
            'smc_grid','smc_grid_reach_current',
            'smc_grid_reach_trl6','smc_grid_curr_diff_trl6',
            'smc_grid_reach_trl9','smc_grid_curr_diff_trl9',
            'hl_route_grid',
            'smc_nodes_with_hl','smc_nodes_per_grid_current',
            'smc_nodes_per_grid_with_hl','smc_nodes_increase_total']
    wide_ppml[cols].to_csv("smart_city_metrics_output_ppml.csv", index=False)
    print(wide_ppml[cols])


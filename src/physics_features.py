import pandas as pd
import numpy as np

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-informed features from raw turbojet sensor data.
    Includes thermodynamic stage efficiencies, corrected parameters,
    interaction features, and degradation-aware cross terms.
    """
    out = df.copy()
    
    # Constants
    gamma_c = 1.4       # Ratio of specific heats (cold air, compressor)
    gamma_t = 1.33      # Ratio of specific heats (hot gas, turbine)
    cp = 1004.0          # J/(kg K)
    R = 287.05           # J/(kg K)
    T_ref = 288.15       # ISA sea-level temperature (K)
    P_ref = 101325.0     # ISA sea-level pressure (Pa)
    
    # Avoid division by zero
    eps = 1e-9
    
    # ========== STAGE 1: COMPRESSOR ==========
    # Pressure & temperature ratios
    out['Compressor_PR'] = out['P2_Pa'] / (out['Pamb_Pa'] + eps)
    out['Compressor_TR'] = out['T2_K'] / (out['Tamb_K'] + eps)
    
    # Isentropic compressor efficiency
    exponent_c = (gamma_c - 1) / gamma_c
    t_ideal_rise_comp = out['Tamb_K'] * (out['Compressor_PR']**exponent_c - 1.0)
    t_actual_rise_comp = out['T2_K'] - out['Tamb_K']
    out['Compressor_Eff'] = t_ideal_rise_comp / (t_actual_rise_comp + eps)
    
    # ========== STAGE 2: COMBUSTOR ==========
    # Temperature rise & efficiency proxy
    out['Combustor_TRise'] = out['T3_K'] - out['T2_K']
    m_dot_proxy_comb = out['P3_Pa'] / np.sqrt(out['T3_K'] + eps)
    out['Combustor_Efficiency_Proxy'] = m_dot_proxy_comb * out['Combustor_TRise'] / (out['FuelFlow_kg_s'] + eps)
    
    # ========== STAGE 3: TURBINE ==========
    # Expansion ratio
    out['Turbine_ER'] = out['P4_Pa'] / (out['P3_Pa'] + eps)
    
    # Temperature drop & isentropic turbine efficiency
    out['Turbine_TDrop'] = out['T3_K'] - out['T4_K']
    exponent_t = (gamma_t - 1) / gamma_t
    t_ideal_drop_turb = out['T3_K'] * (1.0 - out['Turbine_ER']**exponent_t)
    out['Turbine_Eff'] = out['Turbine_TDrop'] / (t_ideal_drop_turb + eps)
    
    # ========== CORRECTED PARAMETERS ==========
    theta = out['Tamb_K'] / T_ref
    delta = out['Pamb_Pa'] / P_ref
    out['Corrected_RPM'] = out['RPM_rev_min'] / np.sqrt(theta + eps)
    out['Corrected_FuelFlow'] = out['FuelFlow_kg_s'] / (delta * np.sqrt(theta) + eps)
    
    # ========== ESTIMATED THRUST PROXY ==========
    m_dot_proxy = out['P3_Pa'] / np.sqrt(out['T3_K'] + eps)
    ve_proxy = np.sqrt(2 * cp * np.maximum(out['Turbine_TDrop'], 0))
    v0 = out['Mach'] * np.sqrt(gamma_c * R * out['Tamb_K'])
    area_proxy = 0.01
    out['Estimated_Thrust_Proxy'] = area_proxy * m_dot_proxy * (ve_proxy - v0)
    
    # ========== NEW: PRESSURE LOSS RATIOS ==========
    out['P3_P2_ratio'] = out['P3_Pa'] / (out['P2_Pa'] + eps)
    out['P4_P3_ratio'] = out['P4_Pa'] / (out['P3_Pa'] + eps)
    
    # ========== NEW: ENTHALPY-BASED FEATURES ==========
    out['Enthalpy_Rise'] = cp * (out['T3_K'] - out['T2_K'])
    out['Enthalpy_Drop'] = cp * (out['T3_K'] - out['T4_K'])
    
    # ========== NEW: CYCLE-BASED INTERACTION FEATURES ==========
    # These capture how efficiency degrades with operational cycles
    if 'Cycle' in out.columns:
        out['Cycle_x_CompEff'] = out['Cycle'] * out['Compressor_Eff']
        out['Cycle_x_TurbEff'] = out['Cycle'] * out['Turbine_Eff']
        out['Cycle_x_CombEff'] = out['Cycle'] * out['Combustor_Efficiency_Proxy']
        out['Cycle_x_PR'] = out['Cycle'] * out['Compressor_PR']
    
    return out

if __name__ == "__main__":
    # Quick test if run directly
    df = pd.read_csv('../data/train.csv')
    df_feat = add_physics_features(df)
    print("Features engineered successfully.")
    print(f"New feature count: {len(df_feat.columns) - len(df.columns)}")
    print(df_feat[['Compressor_Eff', 'Turbine_Eff', 'Estimated_Thrust_Proxy', 'Cycle_x_CompEff']].head())

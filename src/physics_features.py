import pandas as pd
import numpy as np

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-informed features from raw turbojet sensor data.
    """
    out = df.copy()
    
    # Constants
    gamma = 1.4
    cp = 1004.0 # J/(kg K)
    R = 287.05 # J/(kg K)
    T_ref = 288.15
    P_ref = 101325.0
    
    # Avoid division by zero
    eps = 1e-9
    
    # 1. Compressor pressure & temperature ratios
    out['Compressor_PR'] = out['P2_Pa'] / (out['Pamb_Pa'] + eps)
    out['Compressor_TR'] = out['T2_K'] / (out['Tamb_K'] + eps)
    
    # 2. Isentropic compressor efficiency estimate
    exponent = (gamma - 1) / gamma
    t_ideal_rise_comp = out['Tamb_K'] * (out['Compressor_PR']**exponent - 1.0)
    t_actual_rise_comp = out['T2_K'] - out['Tamb_K']
    out['Compressor_Eff'] = t_ideal_rise_comp / (t_actual_rise_comp + eps)
    
    # 3. Combustor temperature rise & normalized
    out['Combustor_TRise'] = out['T3_K'] - out['T2_K']
    out['Combustor_Efficiency_Proxy'] = out['Combustor_TRise'] / (out['FuelFlow_kg_s'] + eps)
    
    # 4. Turbine expansion ratio
    out['Turbine_ER'] = out['P4_Pa'] / (out['P3_Pa'] + eps)
    
    # 5. Turbine temperature drop & isentropic turbine efficiency estimate
    out['Turbine_TDrop'] = out['T3_K'] - out['T4_K']
    t_ideal_drop_turb = out['T3_K'] * (1.0 - out['Turbine_ER']**exponent)
    out['Turbine_Eff'] = out['Turbine_TDrop'] / (t_ideal_drop_turb + eps)
    
    # 6. Corrected RPM and fuel flow
    theta = out['Tamb_K'] / T_ref
    delta = out['Pamb_Pa'] / P_ref
    out['Corrected_RPM'] = out['RPM_rev_min'] / np.sqrt(theta + eps)
    out['Corrected_FuelFlow'] = out['FuelFlow_kg_s'] / (delta * np.sqrt(theta) + eps)
    
    # 7. Estimated thrust proxy
    m_dot_proxy = out['P3_Pa'] / np.sqrt(out['T3_K'] + eps)
    ve_proxy = np.sqrt(2 * cp * np.maximum(out['Turbine_TDrop'], 0))
    v0 = out['Mach'] * np.sqrt(gamma * R * out['Tamb_K'])
    area_proxy = 0.01 
    out['Estimated_Thrust_Proxy'] = area_proxy * m_dot_proxy * (ve_proxy - v0)
    
    # 8. Stage temperature ratios
    out['Combustor_TR'] = out['T3_K'] / (out['T2_K'] + eps)
    out['Turbine_TR'] = out['T4_K'] / (out['T3_K'] + eps)
    out['Overall_TR'] = out['T4_K'] / (out['Tamb_K'] + eps)
    
    # 9. Stage pressure ratios
    out['Combustor_PR'] = out['P3_Pa'] / (out['P2_Pa'] + eps)
    out['Overall_PR'] = out['P4_Pa'] / (out['Pamb_Pa'] + eps)
    
    # 10. Corrected stage pressures and temperatures
    out['T2_corr'] = out['T2_K'] / (theta + eps)
    out['T3_corr'] = out['T3_K'] / (theta + eps)
    out['T4_corr'] = out['T4_K'] / (theta + eps)
    
    out['P2_corr'] = out['P2_Pa'] / (delta + eps)
    out['P3_corr'] = out['P3_Pa'] / (delta + eps)
    out['P4_corr'] = out['P4_Pa'] / (delta + eps)
    
    # 11. Cycle feature (normalized by 30 cycles max)
    out['Cycle_Norm'] = out['Cycle'] / 30.0
    
    return out

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'))
    df_feat = add_physics_features(df)
    print("Features engineered successfully.")
    print(df_feat[['Compressor_Eff', 'Turbine_Eff', 'Estimated_Thrust_Proxy']].head())

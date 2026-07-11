import pandas as pd
import numpy as np

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-informed features from raw turbojet sensor data.
    """
    out = df.copy()
    
    # Constants
    gamma_c = 1.4
    gamma_t = 1.33
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
    exponent_c = (gamma_c - 1) / gamma_c
    t_ideal_rise_comp = out['Tamb_K'] * (out['Compressor_PR']**exponent_c - 1.0)
    t_actual_rise_comp = out['T2_K'] - out['Tamb_K']
    out['Compressor_Eff'] = t_ideal_rise_comp / (t_actual_rise_comp + eps)
    
    # 3. Combustor temperature rise & normalized using mass flow proxy (P3/sqrt(T3))
    out['Combustor_TRise'] = out['T3_K'] - out['T2_K']
    m_dot_proxy = out['P3_Pa'] / np.sqrt(out['T3_K'] + eps)
    out['Combustor_Efficiency_Proxy'] = m_dot_proxy * out['Combustor_TRise'] / (out['FuelFlow_kg_s'] + eps)
    
    # 4. Turbine expansion ratio
    out['Turbine_ER'] = out['P4_Pa'] / (out['P3_Pa'] + eps)
    
    # 5. Turbine temperature drop & isentropic turbine efficiency estimate
    out['Turbine_TDrop'] = out['T3_K'] - out['T4_K']
    exponent_t = (gamma_t - 1) / gamma_t
    t_ideal_drop_turb = out['T3_K'] * (1.0 - out['Turbine_ER']**exponent_t)
    out['Turbine_Eff'] = out['Turbine_TDrop'] / (t_ideal_drop_turb + eps)
    
    # 6. Corrected RPM
    theta = out['Tamb_K'] / T_ref
    delta = out['Pamb_Pa'] / P_ref
    out['Corrected_RPM'] = out['RPM_rev_min'] / np.sqrt(theta + eps)
    out['Corrected_FuelFlow'] = out['FuelFlow_kg_s'] / (delta * np.sqrt(theta) + eps)
    
    # 7. Estimated thrust proxy
    # For a choked nozzle, m_dot is proportional to P3 / sqrt(T3)
    m_dot_proxy = out['P3_Pa'] / np.sqrt(out['T3_K'] + eps)
    # Exhaust velocity proxy: Ve ~ sqrt(2 * cp * T_drop)
    ve_proxy = np.sqrt(2 * cp * np.maximum(out['Turbine_TDrop'], 0))
    # Flight velocity V0 = Mach * sqrt(gamma_c * R * Tamb)
    v0 = out['Mach'] * np.sqrt(gamma_c * R * out['Tamb_K'])
    # F = m_dot * (Ve - V0)
    # We apply a scaling factor purely to bring it to a sensible O(10^4) range for comparison
    # Real mass flow scale requires nozzle area. 
    area_proxy = 0.01 
    out['Estimated_Thrust_Proxy'] = area_proxy * m_dot_proxy * (ve_proxy - v0)
    
    return out

if __name__ == "__main__":
    # Quick test if run directly
    df = pd.read_csv('../data/train.csv')
    df_feat = add_physics_features(df)
    print("Features engineered successfully.")
    print(df_feat[['Compressor_Eff', 'Turbine_Eff', 'Estimated_Thrust_Proxy']].head())

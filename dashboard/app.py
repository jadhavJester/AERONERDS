import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as plto
import plotly.express as px
import joblib
import sys
import os

# Add src to path to import physics_features and degradation_trend
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from physics_features import add_physics_features
from degradation_trend import DegradationTrend

# Page Config
st.set_page_config(page_title="Turbojet Digital Twin", layout="wide", page_icon="✈️")

# Custom CSS for Premium Glassmorphic Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }

    /* Hero Banner with moving gradient */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 40px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 10px 0;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .hero-banner p {
        font-size: 1.15rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
        background: rgba(30, 41, 59, 0.6);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
    }

    /* Engine Grid Layout */
    .engine-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 20px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .engine-card {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .engine-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        background: rgba(15, 23, 42, 0.8);
    }
    .engine-card.healthy {
        border-top: 4px solid #10b981;
    }
    .engine-card.warning {
        border-top: 4px solid #f59e0b;
    }
    .engine-card.critical {
        border-top: 4px solid #ef4444;
    }
    .engine-card-id {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .engine-card-health {
        font-size: 1.95rem;
        font-weight: 800;
        margin: 12px 0 6px 0;
    }
    .engine-card.healthy .engine-card-health { color: #10b981; }
    .engine-card.warning .engine-card-health { color: #f59e0b; }
    .engine-card.critical .engine-card-health { color: #ef4444; }
    .engine-card-rul {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }
    .engine-card-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .engine-card.healthy .engine-card-badge { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .engine-card.warning .engine-card-badge { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .engine-card.critical .engine-card-badge { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Helper function to render premium KPI cards
def render_metric_cards(labels, values, colors=None):
    cols = st.columns(len(labels))
    for idx, (label, val) in enumerate(zip(labels, values)):
        with cols[idx]:
            val_style = ""
            if colors and colors[idx]:
                val_style = f"background: none; -webkit-text-fill-color: {colors[idx]}; color: {colors[idx]};"
            card_html = f"""
            <div class="metric-card">
                <span class="metric-label">{label}</span>
                <span class="metric-value" style="{val_style}">{val}</span>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

# Load data and models helper
@st.cache_data
def load_data_and_models():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Load complete dataset for full lifecycle visualization
    df = pd.read_csv(os.path.join(DATA_DIR, 'turbojet_complete_dataset.csv'))
    df = add_physics_features(df)
    
    # Load all models (stacked pipeline)
    xgb_models = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_models.pkl'))
    lgbm_models = joblib.load(os.path.join(BASE_DIR, 'models', 'lgbm_models.pkl'))
    rf_models = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_models.pkl'))
    stack_models = joblib.load(os.path.join(BASE_DIR, 'models', 'stack_models.pkl'))
    ensemble_models = joblib.load(os.path.join(BASE_DIR, 'models', 'ensemble_models.pkl'))
    metadata = joblib.load(os.path.join(BASE_DIR, 'models', 'features_targets.pkl'))
    
    all_features = metadata['all_features']
    targets = metadata['targets']
    
    # Pre-compute STACKED predictions for the full dataset
    X = df[all_features].values
    for i, target in enumerate(targets):
        # Base model predictions
        pred_xgb = xgb_models[i].predict(X)
        pred_lgbm = lgbm_models[i].predict(X)
        pred_rf = rf_models[i].predict(X)
        
        # Stacked meta-learner prediction
        meta_features = np.column_stack([pred_xgb, pred_lgbm, pred_rf])
        df[target + '_pred'] = stack_models[i].predict(meta_features)
        
        # Uncertainty from bootstrap ensemble
        boot_preds = np.array([m.predict(X) for m in ensemble_models[target]])
        df[target + '_p10'] = np.percentile(boot_preds, 10, axis=0)
        df[target + '_p90'] = np.percentile(boot_preds, 90, axis=0)
        
    return df, xgb_models, ensemble_models, metadata

# Execute Load
try:
    df, xgb_models, ensemble_models, metadata = load_data_and_models()
except Exception as e:
    st.error(f"Error loading models or dataset. Please make sure train_models.py has been run successfully. Details: {e}")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.title("✈_ Turbojet Twin")
    st.markdown("Physics-Informed Digital Twin for Health Monitoring")
    st.markdown("---")
    page = st.radio("Navigation Menu", [
        "🏠 Home (Project Overview)",
        "📊 Fleet Overview",
        "🔍 Engine Detail View",
        "📈 Degradation & RUL",
        "🔬 Model Evaluation"
    ])
    st.markdown("---")
    st.markdown("**Hackathon: HAL x IIT Indore**\nProblem Statement 2")

# --- PAGE 1: HOME (PROJECT OVERVIEW) ---
if page == "🏠 Home (Project Overview)":
    # Vibrant hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>✈️ Four-Stage Turbojet Digital Twin</h1>
        <p>Thermodynamic Physics-Informed Real-Time Diagnostics & Prognostics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Context
    This Digital Twin provides a real-time health-monitoring system for a four-stage turbojet engine.
    By marrying thermodynamic principles (physics-informed machine learning) with advanced regression techniques, we estimate individual component health states, predict performance degradations (Thrust and TSFC), and forecast Remaining Useful Life (RUL) with rigorous confidence intervals.
    
    ---
    
    ### System Architecture & Thermodynamic Framework
    Traditional data-driven models often ignore physical constraints. To enforce conservation of energy and mass, we engineer **thermodynamic features** at each critical stage:
    
    1. **Compressor Stage ($PR_c$, $TR_c$, $\eta_c$):**
       - Pressure Ratio: $PR_c = P_2 / P_{amb}$
       - Temperature Ratio: $TR_c = T_2 / T_{amb}$
       - Isentropic Efficiency Estimate:
         $$\eta_c = \\frac{T_{amb} \\cdot \\left(PR_c^{\\frac{\\gamma-1}{\\gamma}} - 1\\right)}{T_2 - T_{amb}}$$
    
    2. **Combustor Stage ($\Delta T_{comb}$, $\eta_{comb, proxy}$):**
       - Temperature Rise: $\Delta T_{comb} = T_3 - T_2$
       - Combustion Efficiency Proxy: 
         $$\eta_{comb, proxy} = \\frac{T_3 - T_2}{\\dot{m}_f}$$
    
    3. **Turbine Stage ($ER_t$, $\Delta T_{turb}$, $\eta_t$):**
       - Expansion Ratio: $ER_t = P_4 / P_3$
       - Temperature Drop: $\Delta T_{turb} = T_3 - T_4$
       - Isentropic Turbine Efficiency:
         $$\eta_t = \\frac{T_3 - T_4}{T_3 \\cdot \\left(1 - ER_t^{\\frac{\\gamma-1}{\\gamma}}\\right)}$$
         
    4. **Flight & Engine Calibration Parameters:**
       - Corrected RPM ($N_{corr} = N / \sqrt{\\theta}$)
       - Corrected Fuel Flow ($\dot{m}_{f, corr} = \dot{m}_f / (\delta\sqrt{\\theta})$)
       - Estimated Thrust Proxy: 
         $$F_{proxy} = A_p \\cdot \\frac{P_3}{\\sqrt{T_3}} \\left(\\sqrt{2c_p(T_3 - T_4)} - M \\sqrt{\\gamma R T_{amb}}\\right)$$
         
    ---
    
    ### Key Features of this Digital Twin
    * **Multi-Stage Health Surveillance:** Computes specific health factors for Compressor, Combustor, and Turbine stages.
    * **Bootstrap Uncertainty Quantification:** Ensemble of 15 bootstrapped XGBoost models outputs 10th and 90th percentile predictions to capture structural and data-driven uncertainties.
    * **Adaptive Degradation Extrapolation:** Fits linear and exponential trends to historic overall health trajectories to project remaining cycles until the critical maintenance crossover threshold ($0.75$).
    """)

# --- PAGE 2: FLEET OVERVIEW ---
elif page == "📊 Fleet Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Fleet Overview & Grid Status</h1>
        <p>Real-Time Status Grid for Fleet Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate latest cycle state for each of the 10 engines
    latest_cycles = df.loc[df.groupby('EngineID')['Cycle'].idxmax()].copy()
    
    # Fit RUL for each engine to show in the table
    ruls = []
    statuses = []
    for idx, row in latest_cycles.iterrows():
        eng_id = row['EngineID']
        eng_data = df[df['EngineID'] == eng_id].sort_values('Cycle')
        
        dt = DegradationTrend(maintenance_threshold=0.75)
        success = dt.fit(eng_data['Cycle'].values, eng_data['OverallHealth_pred'].values)
        
        if success:
            crossover = dt.estimate_rul_crossover()
            if crossover == np.inf or crossover < 0:
                rul_val = 125  # Clipped limit
            else:
                rul_val = max(0, int(crossover - row['Cycle']))
        else:
            rul_val = 0
            
        ruls.append(rul_val)
        
        # Classify Status
        health = row['OverallHealth_pred']
        if health > 0.90:
            statuses.append("🟢 Healthy")
        elif health > 0.75:
            statuses.append("🟡 Warning")
        else:
            statuses.append("🔴 Critical")
            
    latest_cycles['Estimated RUL (Cycles)'] = ruls
    latest_cycles['Status'] = statuses
    
    critical_count = sum(latest_cycles['OverallHealth_pred'] <= 0.75)
    avg_health = latest_cycles['OverallHealth_pred'].mean()
    avg_rul = np.mean([r for r in ruls if r < 120])
    
    # Render glassmorphic metric cards
    render_metric_cards(
        ["Total Engines", "Critical Engines", "Fleet Avg Health", "Average RUL"],
        ["10", str(critical_count), f"{avg_health:.2%}", f"{avg_rul:.1f} Cycles"],
        [None, "#ef4444" if critical_count > 0 else "#10b981", None, None]
    )
        
    st.markdown("---")
    st.markdown("### Real-Time Fleet Status Grid")
    
    # Render Custom HTML Engine Cards Grid
    grid_html = '<div class="engine-grid">'
    for idx, row in latest_cycles.sort_values('EngineID').iterrows():
        eng_id = int(row['EngineID'])
        health = row['OverallHealth_pred']
        rul_val = int(row['Estimated RUL (Cycles)'])
        status = row['Status']
        
        if "Healthy" in status:
            status_class = "healthy"
            badge_text = "Healthy"
        elif "Warning" in status:
            status_class = "warning"
            badge_text = "Warning"
        else:
            status_class = "critical"
            badge_text = "Critical"
            
        grid_html += f"""<div class="engine-card {status_class}">
<div class="engine-card-id">Engine #{eng_id:02d}</div>
<div class="engine-card-health">{health:.1%}</div>
<div class="engine-card-rul">Cycle {int(row['Cycle'])} • RUL: {rul_val if rul_val < 120 else '>100'} cycles</div>
<span class="engine-card-badge">{badge_text}</span>
</div>"""
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

# --- PAGE 3: ENGINE DETAIL VIEW ---
elif page == "🔍 Engine Detail View":
    st.markdown("""
    <div class="hero-banner">
        <h1>🔍 Engine Health Detail View</h1>
        <p>Interactive Cycle Playback & Stage Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        engine_id = st.selectbox("Select Engine to Analyze", sorted(df['EngineID'].unique()))
    
    engine_df = df[df['EngineID'] == engine_id].sort_values('Cycle').reset_index(drop=True)
    max_cycle = int(engine_df['Cycle'].max())
    
    with col_sel2:
        current_cycle = st.slider("Simulate Cycle Progression", min_value=1, max_value=max_cycle, value=max_cycle)
        
    sim_df = engine_df[engine_df['Cycle'] <= current_cycle]
    current_row = sim_df.iloc[-1]
    
    # Premium metric card rendering for engine stage healths
    render_metric_cards(
        ["Overall Health", "Compressor Health", "Combustor Health", "Turbine Health"],
        [f"{current_row['OverallHealth_pred']:.2%}", f"{current_row['CompressorHealth_pred']:.2%}", f"{current_row['CombustorHealth_pred']:.2%}", f"{current_row['TurbineHealth_pred']:.2%}"],
        [
            "#ef4444" if current_row['OverallHealth_pred'] <= 0.75 else ("#f59e0b" if current_row['OverallHealth_pred'] <= 0.90 else "#10b981"),
            "#ef4444" if current_row['CompressorHealth_pred'] <= 0.75 else ("#f59e0b" if current_row['CompressorHealth_pred'] <= 0.90 else "#10b981"),
            "#ef4444" if current_row['CombustorHealth_pred'] <= 0.75 else ("#f59e0b" if current_row['CombustorHealth_pred'] <= 0.90 else "#10b981"),
            "#ef4444" if current_row['TurbineHealth_pred'] <= 0.75 else ("#f59e0b" if current_row['TurbineHealth_pred'] <= 0.90 else "#10b981")
        ]
    )
        
    st.markdown("---")
    
    # Plot Overall Health with Conformal Band
    fig = plto.Figure()
    # Conformal Shadow Band
    fig.add_trace(plto.Scatter(
        x=sim_df['Cycle'].tolist() + sim_df['Cycle'].tolist()[::-1],
        y=sim_df['OverallHealth_p90'].tolist() + sim_df['OverallHealth_p10'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='10th-90th Conformal Band',
        hoverinfo='skip'
    ))
    # GT
    fig.add_trace(plto.Scatter(
        x=sim_df['Cycle'], y=sim_df['OverallHealth'],
        mode='lines+markers', name='Ground Truth Health',
        line=dict(color='#64748b', dash='dash')
    ))
    # Pred
    fig.add_trace(plto.Scatter(
        x=sim_df['Cycle'], y=sim_df['OverallHealth_pred'],
        mode='lines+markers', name='Surrogate Prediction',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig.update_layout(
        title=f"Overall Health Diagnostics vs Cycle (Engine {engine_id})",
        xaxis_title="Operational Cycles",
        yaxis_title="Overall Health Coeff",
        template="plotly_dark",
        yaxis=dict(range=[0.55, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Component performance charts (Thrust & TSFC)
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_th = plto.Figure()
        fig_th.add_trace(plto.Scatter(x=sim_df['Cycle'], y=sim_df['Thrust_N'], mode='lines', name='GT Thrust', line=dict(color='#64748b', dash='dot')))
        fig_th.add_trace(plto.Scatter(x=sim_df['Cycle'], y=sim_df['Thrust_N_pred'], mode='lines+markers', name='Pred Thrust', line=dict(color='#10b981', width=2.5)))
        fig_th.update_layout(title="Thrust (N) Performance", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_th, use_container_width=True)
        
    with col_g2:
        fig_ts = plto.Figure()
        fig_ts.add_trace(plto.Scatter(x=sim_df['Cycle'], y=sim_df['TSFC_g_N_s'], mode='lines', name='GT TSFC', line=dict(color='#64748b', dash='dot')))
        fig_ts.add_trace(plto.Scatter(x=sim_df['Cycle'], y=sim_df['TSFC_g_N_s_pred'], mode='lines+markers', name='Pred TSFC', line=dict(color='#f59e0b', width=2.5)))
        fig_ts.update_layout(title="TSFC (g/N.s) Efficiency", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_ts, use_container_width=True)
        
    # Stage Parameter Analysis
    st.markdown("### Raw Sensor & Physics-Informed Feature Tracker")
    sensor_options = {
        "Altitude (m)": "Altitude_m",
        "Mach Number": "Mach",
        "RPM (rev/min)": "RPM_rev_min",
        "Fuel Flow (kg/s)": "FuelFlow_kg_s",
        "Compressor PR": "Compressor_PR",
        "Compressor Efficiency": "Compressor_Eff",
        "Combustor Temp Rise (K)": "Combustor_TRise",
        "Combustor Efficiency Proxy": "Combustor_Efficiency_Proxy",
        "Turbine Expansion Ratio": "Turbine_ER",
        "Turbine Efficiency": "Turbine_Eff",
        "Estimated Thrust Proxy": "Estimated_Thrust_Proxy"
    }
    
    selected_sensors = st.multiselect(
        "Select parameters to display on secondary time plot:",
        list(sensor_options.keys()),
        default=["Compressor Efficiency", "Turbine Efficiency"]
    )
    
    if selected_sensors:
        fig_sens = plto.Figure()
        for sens_name in selected_sensors:
            col_key = sensor_options[sens_name]
            fig_sens.add_trace(plto.Scatter(
                x=sim_df['Cycle'], y=sim_df[col_key],
                mode='lines+markers', name=sens_name
            ))
        fig_sens.update_layout(
            title="Sensor & Feature Trajectories",
            xaxis_title="Cycle",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

# --- PAGE 4: DEGRADATION & RUL ---
elif page == "📈 Degradation & RUL":
    st.markdown("""
    <div class="hero-banner">
        <h1>📈 Degradation Forecasting & RUL</h1>
        <p>Adaptive Curve Fitting & Prognostics Prognosis</p>
    </div>
    """, unsafe_allow_html=True)
    
    engine_id = st.selectbox("Select Engine for Forecasting", sorted(df['EngineID'].unique()))
    engine_df = df[df['EngineID'] == engine_id].sort_values('Cycle').reset_index(drop=True)
    
    st.markdown("""
    ### Prognostics Methodology
    To compute the **Remaining Useful Life (RUL)**, we fit linear and exponential degradation curves to the overall health predictions generated so far.
    The model that yields the higher coefficient of determination ($R^2$) is selected to project the trajectory until it crosses the **Maintenance Threshold ($0.75$)**.
    """)
    
    # Fit Trend
    dt = DegradationTrend(maintenance_threshold=0.75)
    success = dt.fit(engine_df['Cycle'].values, engine_df['OverallHealth_pred'].values)
    
    if success:
        crossover = dt.estimate_rul_crossover()
        current_cycle = engine_df['Cycle'].max()
        rul = max(0, int(crossover - current_cycle)) if crossover != np.inf else ">100"
        
        # Glassmorphic KPI cards for prognosis
        render_metric_cards(
            ["Best Fit Model", "Predicted Crossover Cycle", "RUL Remaining"],
            [dt.func_type.capitalize(), f"{crossover:.1f}" if crossover != np.inf else "N/A", f"{rul} Cycles" if type(rul) == int else "Stable"],
            [None, None, "#ef4444" if (type(rul) == int and rul < 30) else "#10b981"]
        )
            
        st.markdown("---")
        
        # Projection plot
        fig_proj = plto.Figure()
        
        # Historical Preds
        fig_proj.add_trace(plto.Scatter(
            x=engine_df['Cycle'], y=engine_df['OverallHealth_pred'],
            mode='markers', name='Predicted Health History',
            marker=dict(color='#3b82f6', size=8)
        ))
        
        # Future Projection
        max_proj_cycle = max(50, int(crossover + 5) if (crossover != np.inf and crossover < 200) else 50)
        future_cycles = np.arange(1, max_proj_cycle + 1)
        extrapolated = dt.predict_health(future_cycles)
        
        fig_proj.add_trace(plto.Scatter(
            x=future_cycles, y=extrapolated,
            mode='lines', name='Prognostic Extrapolation Line',
            line=dict(color='#ef4444', dash='dot', width=2)
        ))
        
        # Maintenance Threshold
        fig_proj.add_hline(
            y=0.75, line_dash="dash", line_color="#ef4444",
            annotation_text="Critical Limit (0.75)", annotation_position="bottom left"
        )
        
        fig_proj.update_layout(
            title=f"Adaptive Health Degradation Trend (Engine {engine_id})",
            xaxis_title="Operational Cycles",
            yaxis_title="Overall Health",
            template="plotly_dark",
            yaxis=dict(range=[0.45, 1.05]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_proj, use_container_width=True)
        
        # Diagnostic report
        if crossover != np.inf:
            st.info(f"**Prognostic Analysis:** Engine {engine_id} is currently showing a clear degradation trend fitted best by the **{dt.func_type}** profile. At the current degradation rate, the engine is estimated to breach the 0.75 safety threshold at cycle **{crossover:.1f}**, leaving **{rul} cycles** of operation before mandatory maintenance is required.")
        else:
            st.success(f"**Prognostic Analysis:** Engine {engine_id} health remains stable or exhibits a negligible downward trend. No maintenance crossover was identified within the forecasting window.")
            
    else:
        st.warning("Not enough history to model degradation trends.")

# --- PAGE 5: MODEL EVALUATION ---
elif page == "🔬 Model Evaluation":
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 Machine Learning Model Evaluation</h1>
        <p>Cross-Validation Performance & Feature Importance Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Cross-Validation & Modeling Framework
    To prevent data leakage across cycles of the same engine, we validated all models using **5-Fold GroupKFold Cross-Validation** grouped by `EngineID`.
    
    Below is the comparison of our **Stacked Ensemble** (XGBoost + LightGBM + Random Forest → Ridge Meta-Learner) against individual base learners and a **Ridge Regression** baseline fitted on physics features.
    """)
    
    # Load and render metrics table
    metrics_data = [
        {"Target": "CompressorHealth", "Model": "XGBoost", "R2": 0.9163, "MAE": 0.0154, "RMSE": 0.0211},
        {"Target": "CompressorHealth", "Model": "LightGBM", "R2": 0.9263, "MAE": 0.0146, "RMSE": 0.0198},
        {"Target": "CompressorHealth", "Model": "RandomForest", "R2": 0.9359, "MAE": 0.0133, "RMSE": 0.0185},
        {"Target": "CompressorHealth", "Model": "Ridge (Physics)", "R2": 0.9106, "MAE": 0.0161, "RMSE": 0.0218},
        {"Target": "CompressorHealth", "Model": "⭐ Stacked Ensemble", "R2": 0.9227, "MAE": 0.0148, "RMSE": 0.0203},
        
        {"Target": "CombustorHealth", "Model": "XGBoost", "R2": 0.7627, "MAE": 0.0090, "RMSE": 0.0130},
        {"Target": "CombustorHealth", "Model": "LightGBM", "R2": 0.7773, "MAE": 0.0088, "RMSE": 0.0126},
        {"Target": "CombustorHealth", "Model": "RandomForest", "R2": 0.7794, "MAE": 0.0085, "RMSE": 0.0125},
        {"Target": "CombustorHealth", "Model": "Ridge (Physics)", "R2": 0.7615, "MAE": 0.0093, "RMSE": 0.0130},
        {"Target": "CombustorHealth", "Model": "⭐ Stacked Ensemble", "R2": 0.7205, "MAE": 0.0098, "RMSE": 0.0141},
        
        {"Target": "TurbineHealth", "Model": "XGBoost", "R2": 0.8034, "MAE": 0.0170, "RMSE": 0.0229},
        {"Target": "TurbineHealth", "Model": "LightGBM", "R2": 0.7914, "MAE": 0.0174, "RMSE": 0.0235},
        {"Target": "TurbineHealth", "Model": "RandomForest", "R2": 0.8088, "MAE": 0.0165, "RMSE": 0.0225},
        {"Target": "TurbineHealth", "Model": "Ridge (Physics)", "R2": 0.8241, "MAE": 0.0168, "RMSE": 0.0216},
        {"Target": "TurbineHealth", "Model": "⭐ Stacked Ensemble", "R2": 0.7992, "MAE": 0.0173, "RMSE": 0.0231},
        
        {"Target": "OverallHealth", "Model": "XGBoost", "R2": 0.9273, "MAE": 0.0097, "RMSE": 0.0138},
        {"Target": "OverallHealth", "Model": "LightGBM", "R2": 0.9351, "MAE": 0.0092, "RMSE": 0.0130},
        {"Target": "OverallHealth", "Model": "RandomForest", "R2": 0.9436, "MAE": 0.0081, "RMSE": 0.0121},
        {"Target": "OverallHealth", "Model": "Ridge (Physics)", "R2": 0.9323, "MAE": 0.0093, "RMSE": 0.0133},
        {"Target": "OverallHealth", "Model": "⭐ Stacked Ensemble", "R2": 0.9362, "MAE": 0.0089, "RMSE": 0.0129},
        
        {"Target": "Thrust_N", "Model": "XGBoost", "R2": 0.9801, "MAE": 1854.12, "RMSE": 2424.66},
        {"Target": "Thrust_N", "Model": "LightGBM", "R2": 0.9848, "MAE": 1655.84, "RMSE": 2114.16},
        {"Target": "Thrust_N", "Model": "RandomForest", "R2": 0.9638, "MAE": 2572.81, "RMSE": 3264.76},
        {"Target": "Thrust_N", "Model": "Ridge (Physics)", "R2": 0.9307, "MAE": 3409.05, "RMSE": 4520.23},
        {"Target": "Thrust_N", "Model": "⭐ Stacked Ensemble", "R2": 0.9857, "MAE": 1592.40, "RMSE": 2054.80},
        
        {"Target": "TSFC_g_N_s", "Model": "XGBoost", "R2": 0.8654, "MAE": 0.0016, "RMSE": 0.0023},
        {"Target": "TSFC_g_N_s", "Model": "LightGBM", "R2": 0.8849, "MAE": 0.0016, "RMSE": 0.0022},
        {"Target": "TSFC_g_N_s", "Model": "RandomForest", "R2": 0.9762, "MAE": 0.0008, "RMSE": 0.0010},
        {"Target": "TSFC_g_N_s", "Model": "Ridge (Physics)", "R2": 0.9792, "MAE": 0.0006, "RMSE": 0.0009},
        {"Target": "TSFC_g_N_s", "Model": "⭐ Stacked Ensemble", "R2": 0.9770, "MAE": 0.0008, "RMSE": 0.0010}
    ]
    
    m_df = pd.DataFrame(metrics_data)
    st.table(m_df.set_index(["Target", "Model"]))
    
    st.markdown("---")
    
    # Feature Importance Plot
    st.subheader("Feature Importance Plot (XGBoost Surrogate)")
    target_sel = st.selectbox("Select Target Variable for Feature Importance:", metadata['targets'])
    
    target_idx = metadata['targets'].index(target_sel)
    model = xgb_models[target_idx]
    
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': metadata['all_features'],
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(
        feat_df, x='Importance', y='Feature', orientation='h',
        title=f"Top Features driving {target_sel}",
        template="plotly_dark",
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

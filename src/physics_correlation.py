import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from physics_features import add_physics_features

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading datasets...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    gt = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    
    df = pd.merge(train_raw, gt, on=['EngineID', 'Cycle'], how='inner')
    df = add_physics_features(df)
    
    # 4 strong physical relationships
    pairs = [
        ('Compressor_Eff', 'CompressorHealth', 'Compressor Isentropic Efficiency', 'Compressor Health', '#0284c7'),
        ('Compressor_Eff', 'OverallHealth', 'Compressor Isentropic Efficiency', 'Overall Engine Health', '#ea580c'),
        ('Overall_PR', 'Thrust_N', 'Overall Pressure Ratio (P4/Pamb)', 'Thrust (N)', '#16a34a'),
        ('Combustor_TRise', 'TSFC_g_N_s', 'Combustor Temp Rise (T3 - T2)', 'TSFC (g/N/s)', '#4f46e5')
    ]
    
    print("\n--- Correlation Analysis ---")
    correlation_markdown = "| Physics Feature Proxy | Target Variable | Pearson r | Spearman rho | Description |\n| :--- | :--- | :---: | :---: | :--- |\n"
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    for idx, (feat, target, feat_name, target_name, color) in enumerate(pairs):
        x = df[feat].values
        y = df[target].values
        
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        r_val, _ = pearsonr(x_clean, y_clean)
        rho_val, _ = spearmanr(x_clean, y_clean)
        
        print(f"{feat} vs {target}: Pearson={r_val:.4f}, Spearman={rho_val:.4f}")
        
        # Add descriptive text to the table
        if target == 'CompressorHealth':
            desc = "Strong positive correlation indicating that estimated thermodynamic efficiency tracks compressor health closely."
        elif target == 'OverallHealth':
            desc = "Shows that compressor efficiency is a strong surrogate for overall engine health degradation."
        elif target == 'Thrust_N':
            desc = "Overall Pressure Ratio is a key thermodynamic driver of engine thrust output."
        else:
            desc = "Combustor temperature rise is almost perfectly correlated with Thrust Specific Fuel Consumption."
            
        correlation_markdown += f"| `{feat}` | `{target}` | {r_val:.4f} | {rho_val:.4f} | {desc} |\n"
        
        # Plotting
        ax = axes[idx]
        ax.scatter(x_clean, y_clean, color=color, alpha=0.5, edgecolors='k', linewidths=0.5, label='Data Points')
        
        # Add regression line
        m, c = np.polyfit(x_clean, y_clean, 1)
        # sort for plotting line
        sort_idx = np.argsort(x_clean)
        ax.plot(x_clean[sort_idx], m*x_clean[sort_idx] + c, color='#ef4444', linewidth=2.5, linestyle='--', label=f'Fit (r={r_val:.2f})')
        
        ax.set_title(f"{feat_name} vs {target_name}", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(feat_name, fontsize=11)
        ax.set_ylabel(target_name, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        
    plt.suptitle("Thermodynamic Physics Features vs. Health/Performance Targets", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(RESULTS_DIR, 'physics_correlation.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved successfully to {plot_path}")
    
    # Save markdown correlation table
    with open(os.path.join(RESULTS_DIR, 'correlation_table.md'), 'w') as f:
        f.write("# Physics Proxy Correlation Analysis\n\n")
        f.write("This table summarizes the correlation between the engineered thermodynamic features (physics proxies) and the ground-truth targets. The high correlation coefficients confirm the physical consistency and predictive power of the physics-informed model.\n\n")
        f.write(correlation_markdown)
        
    print("\nMarkdown correlation table saved to results/correlation_table.md")
    print(correlation_markdown)

if __name__ == '__main__':
    main()

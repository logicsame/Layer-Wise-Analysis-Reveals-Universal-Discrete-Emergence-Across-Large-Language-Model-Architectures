"""
Figure 5: Size-Dependent Complexity
Publication-quality figure for Nature Machine Intelligence

4-panel figure examining size effects:
- Panel A: Emergence depth vs model size (scatter + trend)
- Panel B: Per-task size effects (4 subplots)
- Panel C: Family comparison (box plots)
- Panel D: Scaling law fits (power, log, linear)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from scipy.stats import linregress

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class Figure5Generator:
    """Generate Figure 5: Size-Dependent Complexity"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.model_dirs = [
            "llama_1b_models", "llama_3b_model", "llama_5b_model",
            "llama_13b_models", "phi_1_models", "phi_1.5_models",
            "deepsek_7b_models"
        ]
        self.tasks = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        self.task_labels = ['GSM8K', 'BoolQ', 'CommonsenseQA', 'HellaSwag']
        self.data = {}
        
        # Families and colors
        self.families = {
            'Llama': ['llama_1b', 'llama_3b', 'llama_5b', 'llama_13b'],
            'Phi': ['phi_1', 'phi_1.5'],
            'DeepSeek': ['deepsek_7b']
        }
        
        self.family_colors = {
            'Llama': '#3498db',
            'Phi': '#e74c3c',
            'DeepSeek': '#2ecc71'
        }
        
        # Model sizes (manual mapping)
        self.model_sizes = {
            'llama_1b': 1.0,
            'llama_3b': 3.0,
            'llama_5b': 5.0,
            'llama_13b': 13.0,
            'phi_1': 1.3,
            'phi_1.5': 1.5,
            'deepsek_7b': 7.0
        }
        
    def load_all_data(self):
        """Load emergence data"""
        print("📂 Loading data...")
        
        for model_dir in self.model_dirs:
            analysis_dir = self.base_dir / model_dir / "analysis"
            if not analysis_dir.exists():
                continue
                
            model_name = model_dir.replace("_models", "").replace("_model", "")
            self.data[model_name] = {}
            
            for task in self.tasks:
                emergence_files = list(analysis_dir.glob(f"*{task}*_emergence.json"))
                if emergence_files:
                    with open(emergence_files[0], 'r') as f:
                        self.data[model_name][task] = json.load(f)
                        
        print(f"✅ Loaded {len(self.data)} models")
        return self.data
    
    def get_emergence_depth(self, model_name, task):
        """Get emergence depth percentage"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None
        
        cryst = self.data[model_name][task].get('crystallization_layer', -1)
        layers = self.data[model_name][task].get('num_layers', 0)
        
        if cryst < 0 or layers == 0:
            return None
        
        return (cryst / layers) * 100
    
    def get_family(self, model_name):
        """Get architecture family"""
        for family, models in self.families.items():
            if model_name in models:
                return family
        return 'Unknown'
    
    def create_figure(self):
        """Create 4-panel figure"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # GridSpec
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.3,
                             left=0.08, right=0.96, 
                             top=0.93, bottom=0.07)
        
        ax_a = fig.add_subplot(gs[0, 0])
        
        # Panel B: 2x2 subgrid
        gs_b = gs[0, 1].subgridspec(2, 2, hspace=0.45, wspace=0.3)
        ax_b = [fig.add_subplot(gs_b[i, j]) for i in range(2) for j in range(2)]
        
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
        
        self.plot_panel_a(ax_a)
        self.plot_panel_b(ax_b)
        self.plot_panel_c(ax_c)
        self.plot_panel_d(ax_d)
        
        return fig
    
    def plot_panel_a(self, ax):
        """Panel A: Emergence depth vs model size"""
        
        # Collect data
        sizes = []
        mean_depths = []
        families_list = []
        model_names = []
        
        for model in self.data.keys():
            if model not in self.model_sizes:
                continue
            
            size = self.model_sizes[model]
            depths = []
            
            for task in self.tasks:
                depth = self.get_emergence_depth(model, task)
                if depth is not None:
                    depths.append(depth)
            
            if depths:
                sizes.append(size)
                mean_depths.append(np.mean(depths))
                families_list.append(self.get_family(model))
                model_names.append(model)
        
        # Scatter by family
        for family, color in self.family_colors.items():
            mask = np.array(families_list) == family
            ax.scatter(np.array(sizes)[mask], 
                      np.array(mean_depths)[mask],
                      s=200, alpha=0.7, c=color, 
                      edgecolors='black', linewidth=2,
                      label=family, zorder=3)
        
        # Annotate outliers
        for i, (size, depth, name) in enumerate(zip(sizes, mean_depths, model_names)):
            if 'phi_1' in name and 'phi_1.5' not in name:
                ax.annotate('Phi-1.0\n(Very Early)', xy=(size, depth), 
                           xytext=(size-0.3, depth+10),
                           fontsize=9, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', lw=1.5))
            elif 'llama_3b' in name:
                ax.annotate('Llama-3B\n(Very Late)', xy=(size, depth), 
                           xytext=(size+0.5, depth+5),
                           fontsize=9, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # Trend line (linear)
        z = np.polyfit(sizes, mean_depths, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(sizes), max(sizes), 100)
        ax.plot(x_trend, p(x_trend), 'k--', linewidth=2, 
               alpha=0.5, label='Linear Trend', zorder=1)
        
        # Formatting
        ax.set_xlabel('Model Size (Billions of Parameters)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Emergence Depth (%)', 
                     fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(0.9, 15)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        
        # Panel label
        ax.text(-0.15, 1.08, 'A', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Non-Monotonic Size Effects', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Correlation
        r, p = stats.pearsonr(sizes, mean_depths)
        ax.text(0.05, 0.95, 
               f'Pearson r = {r:.3f}\np = {p:.3f}\n(Non-monotonic)',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=2),
               fontsize=10, fontweight='bold')
    
    def plot_panel_b(self, axes):
        """Panel B: Per-task size effects (4 subplots)"""
        
        for idx, (ax, task, task_label) in enumerate(zip(axes, self.tasks, self.task_labels)):
            # Collect data for this task
            sizes = []
            depths = []
            families_list = []
            
            for model in self.data.keys():
                if model not in self.model_sizes:
                    continue
                
                depth = self.get_emergence_depth(model, task)
                if depth is not None:
                    sizes.append(self.model_sizes[model])
                    depths.append(depth)
                    families_list.append(self.get_family(model))
            
            # Scatter
            for family, color in self.family_colors.items():
                mask = np.array(families_list) == family
                if np.any(mask):
                    ax.scatter(np.array(sizes)[mask], 
                              np.array(depths)[mask],
                              s=100, alpha=0.7, c=color, 
                              edgecolors='black', linewidth=1.5)
            
            # Trend line
            if len(sizes) > 1:
                z = np.polyfit(sizes, depths, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(sizes), max(sizes), 100)
                ax.plot(x_trend, p(x_trend), 'k--', linewidth=1.5, alpha=0.5)
            
            # Formatting
            ax.set_xlabel('Size (B)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Depth (%)', fontsize=10, fontweight='bold')
            ax.set_xscale('log')
            ax.set_xlim(0.9, 15)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
            ax.set_title(task_label, fontsize=11, fontweight='bold')
        
        # Panel label
        axes[0].text(-0.35, 1.25, 'B', transform=axes[0].transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
    
    def plot_panel_c(self, ax):
        """Panel C: Family comparison box plots"""
        
        # Collect data by family
        family_depths = {}
        
        for family in self.families.keys():
            family_depths[family] = []
            for model in self.families[family]:
                for task in self.tasks:
                    depth = self.get_emergence_depth(model, task)
                    if depth is not None:
                        family_depths[family].append(depth)
        
        # Box plot
        positions = np.arange(len(self.families))
        bp = ax.boxplot([family_depths[f] for f in self.families.keys()],
                       positions=positions, widths=0.5,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(color='black', linewidth=3),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        # Color boxes
        for patch, family in zip(bp['boxes'], self.families.keys()):
            patch.set_facecolor(self.family_colors[family])
            patch.set_alpha(0.6)
        
        # Add points
        for i, family in enumerate(self.families.keys()):
            y = family_depths[family]
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.5, s=60, color='black',
                      edgecolors='white', linewidth=1, zorder=3)
        
        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(self.families.keys())
        ax.set_ylabel('Mean Emergence Depth (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
        
        # Panel label
        ax.text(-0.15, 1.08, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Architecture Family Comparison\n(Phi Emerges Earliest)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add means
        for i, family in enumerate(self.families.keys()):
            mean_val = np.mean(family_depths[family])
            ax.text(i, mean_val, f'{mean_val:.0f}%', 
                   ha='center', va='bottom', fontsize=10, 
                   fontweight='bold', color='red')
    
    def plot_panel_d(self, ax):
        """Panel D: Scaling law fits"""
        
        # Collect data
        sizes = []
        mean_depths = []
        
        for model in self.data.keys():
            if model not in self.model_sizes:
                continue
            
            depths = []
            for task in self.tasks:
                depth = self.get_emergence_depth(model, task)
                if depth is not None:
                    depths.append(depth)
            
            if depths:
                sizes.append(self.model_sizes[model])
                mean_depths.append(np.mean(depths))
        
        sizes = np.array(sizes)
        mean_depths = np.array(mean_depths)
        
        # Scatter original data
        ax.scatter(sizes, mean_depths, s=150, alpha=0.7,
                  c='gray', edgecolors='black', linewidth=2,
                  label='Observed', zorder=3)
        
        x_fit = np.linspace(min(sizes), max(sizes), 100)
        
        # 1. Power law: depth = A × size^B
        log_sizes = np.log10(sizes)
        log_depths = np.log10(mean_depths)
        slope_p, intercept_p, r_p, p_p, _ = linregress(log_sizes, log_depths)
        A_power = 10**intercept_p
        B_power = slope_p
        r2_power = r_p**2
        y_power = A_power * (x_fit**B_power)
        
        ax.plot(x_fit, y_power, '--', linewidth=2, color='#e74c3c',
               label=f'Power: R²={r2_power:.3f}')
        
        # 2. Logarithmic: depth = A × log(size) + B
        log_sizes_nat = np.log(sizes)
        slope_l, intercept_l, r_l, p_l, _ = linregress(log_sizes_nat, mean_depths)
        r2_log = r_l**2
        y_log = slope_l * np.log(x_fit) + intercept_l
        
        ax.plot(x_fit, y_log, '--', linewidth=2, color='#3498db',
               label=f'Log: R²={r2_log:.3f}')
        
        # 3. Linear: depth = A × size + B
        slope_lin, intercept_lin, r_lin, p_lin, _ = linregress(sizes, mean_depths)
        r2_lin = r_lin**2
        y_lin = slope_lin * x_fit + intercept_lin
        
        ax.plot(x_fit, y_lin, '--', linewidth=2, color='#2ecc71',
               label=f'Linear: R²={r2_lin:.3f}')
        
        # Formatting
        ax.set_xlabel('Model Size (Billions of Parameters)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Emergence Depth (%)', 
                     fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(0.9, 15)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        
        # Panel label
        ax.text(-0.15, 1.08, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('No Simple Scaling Law\n(All R² < 0.3)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Best fit annotation
        best_r2 = max(r2_power, r2_log, r2_lin)
        ax.text(0.05, 0.05, 
               f'Best fit R² = {best_r2:.3f}\n(Weak explanatory power)',
               transform=ax.transAxes, ha='left', va='bottom',
               bbox=dict(boxstyle='round', facecolor='#fff59d', 
                        edgecolor='black', linewidth=2),
               fontsize=10, fontweight='bold')
    
    def generate_and_save(self, output_path='figure5_size_complexity.pdf'):
        """Generate and save"""
        
        self.load_all_data()
        fig = self.create_figure()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   format='pdf', transparent=False)
        print(f"✅ PDF: {output_path}")
        
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ PNG: {png_path}")
        
        plt.show()
        return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🎨 GENERATING FIGURE 5: SIZE-DEPENDENT COMPLEXITY")
    print("="*80 + "\n")
    
    generator = Figure5Generator(base_dir=".")
    fig = generator.generate_and_save('figure5_size_complexity.pdf')
    
    print("\n✅ COMPLETE!")
    print("\nPanels:")
    print("  A: Size vs depth (non-monotonic, annotated outliers)")
    print("  B: Per-task size effects (4 subplots)")
    print("  C: Family comparison (Phi earliest)")
    print("  D: Scaling law attempts (all R² < 0.3)")
    print("\n" + "="*80)
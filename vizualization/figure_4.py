"""
Figure 4: Cross-Architecture Validation
Publication-quality figure for Nature Machine Intelligence

4-panel figure demonstrating architectural robustness:
- Panel A: Emergence depth by architecture family (grouped bars)
- Panel B: Layer count vs emergence depth (scatter)
- Panel C: Normalized depth heatmap (7 models × 4 tasks)
- Panel D: Summary metrics table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import spearmanr

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class Figure4Generator:
    """Generate Figure 4: Cross-Architecture Validation"""
    
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
        
        # Architecture families
        self.families = {
            'Llama': ['llama_1b', 'llama_3b', 'llama_5b', 'llama_13b'],
            'Phi': ['phi_1', 'phi_1.5'],
            'DeepSeek': ['deepsek_7b']
        }
        
        # Family colors
        self.family_colors = {
            'Llama': '#3498db',
            'Phi': '#e74c3c',
            'DeepSeek': '#2ecc71'
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
    
    def get_absolute_emergence_layer(self, model_name, task):
        """Get absolute emergence layer"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None
        return self.data[model_name][task].get('crystallization_layer', -1)
    
    def get_num_layers(self, model_name, task):
        """Get total number of layers"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None
        return self.data[model_name][task].get('num_layers', 0)
    
    def get_family(self, model_name):
        """Get architecture family for a model"""
        for family, models in self.families.items():
            if model_name in models:
                return family
        return 'Unknown'
    
    def create_figure(self):
        """Create 4-panel figure"""
        
        # Large figure
        fig = plt.figure(figsize=(16, 12))
        
        # GridSpec
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.3,
                             left=0.08, right=0.96, 
                             top=0.93, bottom=0.07)
        
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
        
        self.plot_panel_a(ax_a)
        self.plot_panel_b(ax_b)
        self.plot_panel_c(ax_c)
        self.plot_panel_d(ax_d)
        
        return fig
    
    def plot_panel_a(self, ax):
        """Panel A: Grouped bar chart by architecture family"""
        
        # Organize data by family and task
        family_task_data = {}
        
        for family_name, models in self.families.items():
            family_task_data[family_name] = {}
            for task in self.tasks:
                depths = []
                for model in models:
                    depth = self.get_emergence_depth(model, task)
                    if depth is not None:
                        depths.append(depth)
                family_task_data[family_name][task] = depths
        
        # Create grouped bar chart
        x = np.arange(len(self.tasks))
        width = 0.25
        
        for i, (family, color) in enumerate(self.family_colors.items()):
            if family not in family_task_data:
                continue
            
            means = [np.mean(family_task_data[family][task]) if family_task_data[family][task] else 0 
                    for task in self.tasks]
            stds = [np.std(family_task_data[family][task]) if len(family_task_data[family][task]) > 1 else 0
                   for task in self.tasks]
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, 
                         label=f'{family} (n={len(self.families[family])})',
                         color=color, alpha=0.7, edgecolor='black', 
                         linewidth=1.5, capsize=5, error_kw={'linewidth': 2})
        
        # Formatting
        ax.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax.set_ylabel('Emergence Depth (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.task_labels)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        # Panel label
        ax.text(-0.12, 1.08, 'A', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Emergence Depth by Architecture Family', 
                    fontsize=13, fontweight='bold', pad=15)
    
    def plot_panel_b(self, ax):
        """Panel B: Layer count vs absolute emergence layer"""
        
        # Collect data
        layer_counts = []
        emergence_layers = []
        families = []
        
        for model in self.data.keys():
            for task in self.tasks:
                num_layers = self.get_num_layers(model, task)
                emerg_layer = self.get_absolute_emergence_layer(model, task)
                
                if num_layers and emerg_layer and emerg_layer >= 0:
                    layer_counts.append(num_layers)
                    emergence_layers.append(emerg_layer)
                    families.append(self.get_family(model))
        
        # Scatter plot colored by family
        for family, color in self.family_colors.items():
            mask = np.array(families) == family
            ax.scatter(np.array(layer_counts)[mask], 
                      np.array(emergence_layers)[mask],
                      s=120, alpha=0.7, c=color, 
                      edgecolors='black', linewidth=1.5,
                      label=family)
        
        # Formatting
        ax.set_xlabel('Total Layers in Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Emergence Layer', fontsize=12, fontweight='bold')
        ax.set_xlim(15, 45)
        ax.set_ylim(-2, 45)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        # Panel label
        ax.text(-0.12, 1.08, 'B', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Layer Count vs Emergence Depth\n(No Simple Relationship)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Correlation annotation
        from scipy.stats import pearsonr
        r, p = pearsonr(layer_counts, emergence_layers)
        ax.text(0.95, 0.05, 
               f'Pearson r = {r:.3f}\np = {p:.3f}',
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=2),
               fontsize=10, fontweight='bold')
    
    def plot_panel_c(self, ax):
        """Panel C: Heatmap of normalized depths"""
        
        # Create matrix: models × tasks
        models = list(self.data.keys())
        matrix = np.zeros((len(models), len(self.tasks)))
        
        for i, model in enumerate(models):
            for j, task in enumerate(self.tasks):
                depth = self.get_emergence_depth(model, task)
                matrix[i, j] = depth if depth is not None else np.nan
        
        # Create heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', 
                      vmin=0, vmax=100, interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(self.tasks)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(self.task_labels)
        
        # Model labels with family indicators
        model_labels = []
        for model in models:
            family = self.get_family(model)
            label = f"{model.replace('_', '-').upper()} ({family[0]})"
            model_labels.append(label)
        ax.set_yticklabels(model_labels)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(self.tasks)):
                value = matrix[i, j]
                if not np.isnan(value):
                    color = 'white' if value > 50 else 'black'
                    ax.text(j, i, f'{value:.0f}', 
                           ha='center', va='center', 
                           color=color, fontsize=9, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Emergence Depth (%)', rotation=270, labelpad=20, fontsize=11)
        
        # Panel label
        ax.text(-0.12, 1.08, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Normalized Emergence Depths\n(Task Patterns Preserved)', 
                    fontsize=13, fontweight='bold', pad=15)
    
    def plot_panel_d(self, ax):
        """Panel D: Summary metrics table"""
        
        ax.axis('off')
        
        # Panel label
        ax.text(-0.12, 1.08, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        
        # Title
        ax.text(0.5, 0.95, 'Key Metrics Across Architecture Families', 
               ha='center', va='top', fontsize=13, fontweight='bold',
               transform=ax.transAxes)
        
        # Calculate metrics for each family
        table_data = []
        
        for family, models in self.families.items():
            # 1. Abrupt fraction
            abrupt_count = 0
            total_count = 0
            for model in models:
                for task in self.tasks:
                    if model in self.data and task in self.data[model]:
                        layer_stats = self.data[model][task].get('layer_statistics', [])
                        if layer_stats:
                            accs = [s['accuracy'] for s in layer_stats]
                            diffs = np.diff(accs)
                            max_jump = np.max(diffs) if len(diffs) > 0 else 0
                            if max_jump > 0.2:
                                abrupt_count += 1
                            total_count += 1
            
            abrupt_frac = (abrupt_count / total_count * 100) if total_count > 0 else 0
            
            # 2. Task hierarchy correlation
            task_order = ['hellaswag', 'gsm8k', 'commonsense_qa', 'boolq']
            depths_by_task = {task: [] for task in task_order}
            for model in models:
                for task in task_order:
                    depth = self.get_emergence_depth(model, task)
                    if depth is not None:
                        depths_by_task[task].append(depth)
            
            mean_depths = [np.mean(depths_by_task[task]) if depths_by_task[task] else 0 
                          for task in task_order]
            ranks_expected = [1, 2, 3, 4]
            ranks_observed = [sorted(mean_depths).index(d) + 1 for d in mean_depths]
            rho, _ = spearmanr(ranks_expected, ranks_observed)
            
            # 3. Final accuracy CV
            final_accs = []
            for model in models:
                for task in self.tasks:
                    if model in self.data and task in self.data[model]:
                        acc = self.data[model][task].get('overall_accuracy')
                        if acc is not None:
                            final_accs.append(acc * 100)
            
            cv = (np.std(final_accs) / np.mean(final_accs) * 100) if final_accs else 0
            
            table_data.append([family, f'{abrupt_frac:.0f}%', f'{rho:.2f}', f'{cv:.2f}%'])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Family', 'Abrupt\nFraction', 'Task\nHierarchy ρ', 'Final Acc\nCV (%)'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.7])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        
        # Style rows by family
        for i, (family, _) in enumerate(self.families.items(), 1):
            color = self.family_colors[family]
            for j in range(4):
                cell = table[(i, j)]
                cell.set_facecolor(color)
                cell.set_alpha(0.3)
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                if j == 0:
                    cell.set_text_props(weight='bold')
    
    def generate_and_save(self, output_path='figure4_cross_architecture.pdf'):
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
    print("🎨 GENERATING FIGURE 4: CROSS-ARCHITECTURE VALIDATION")
    print("="*80 + "\n")
    
    generator = Figure4Generator(base_dir=".")
    fig = generator.generate_and_save('figure4_cross_architecture.pdf')
    
    print("\n✅ COMPLETE!")
    print("\nPanels:")
    print("  A: Emergence depth by family (Llama, Phi, DeepSeek)")
    print("  B: Layer count vs emergence (no simple relationship)")
    print("  C: Normalized depth heatmap (7 models × 4 tasks)")
    print("  D: Summary metrics table (validates universality)")
    print("\n" + "="*80)
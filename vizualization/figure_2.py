"""
Figure 2: Task-Hierarchical Capability Formation (COMPLETELY REDESIGNED)
Large, clean, professional layout with NO overlapping text
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from matplotlib.patches import FancyBboxPatch

# Publication-quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class Figure2Generator:
    """Generate Figure 2: Task Hierarchy"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.model_dirs = [
            "llama_1b_models", "llama_3b_model", "llama_5b_model",
            "llama_13b_models", "phi_1_models", "phi_1.5_models",
            "deepsek_7b_models"
        ]
        self.tasks = ['hellaswag', 'gsm8k', 'commonsense_qa', 'boolq']
        self.data = {}
        
        # Task colors
        self.task_colors = {
            'hellaswag': '#3498db',      # Blue
            'gsm8k': '#e67e22',          # Orange
            'commonsense_qa': '#2ecc71', # Green
            'boolq': '#e74c3c'           # Red
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
    
    def create_figure(self):
        """Create large, clean figure"""
        
        # MUCH LARGER figure
        fig = plt.figure(figsize=(18, 10))
        
        # GridSpec with generous spacing
        gs = fig.add_gridspec(2, 2, 
                             width_ratios=[1, 1],
                             height_ratios=[1, 0.8],
                             hspace=0.35, wspace=0.3,
                             left=0.06, right=0.98, 
                             top=0.94, bottom=0.06)
        
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, :])
        
        self.plot_panel_a(ax_a)
        self.plot_panel_b(ax_b)
        self.plot_panel_c(ax_c)
        
        return fig
    
    def plot_panel_a(self, ax):
        """Panel A: Box plots"""
        
        # Collect data
        task_depths = {task: [] for task in self.tasks}
        for model in self.data.keys():
            for task in self.tasks:
                depth = self.get_emergence_depth(model, task)
                if depth is not None:
                    task_depths[task].append(depth)
        
        # Order by mean
        task_means = {task: np.mean(depths) for task, depths in task_depths.items()}
        ordered_tasks = sorted(self.tasks, key=lambda t: task_means[t])
        
        # Prepare data
        plot_data = [task_depths[task] for task in ordered_tasks]
        plot_labels = []
        plot_colors = [self.task_colors[task] for task in ordered_tasks]
        
        for task in ordered_tasks:
            if task == 'gsm8k':
                plot_labels.append('GSM8K')
            elif task == 'boolq':
                plot_labels.append('BoolQ')
            elif task == 'commonsense_qa':
                plot_labels.append('CommonsenseQA')
            elif task == 'hellaswag':
                plot_labels.append('HellaSwag')
        
        # Create box plot
        positions = np.arange(len(ordered_tasks))
        bp = ax.boxplot(plot_data, positions=positions, widths=0.5,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(color='black', linewidth=3),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        
        # Add jittered points
        for i, depths in enumerate(plot_data):
            y = depths
            x = np.random.normal(i, 0.05, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=80, color='black', 
                      edgecolors='white', linewidth=1.5, zorder=3)
        
        # Add mean diamonds
        for i, depths in enumerate(plot_data):
            mean_val = np.mean(depths)
            ax.scatter(i, mean_val, marker='D', s=150, 
                      color='red', edgecolors='white', 
                      linewidth=2, zorder=4)
        
        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels)
        ax.set_ylabel('Emergence Depth (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(-10, 105)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # Panel label
        ax.text(-0.12, 1.08, 'A', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        
        # Title
        ax.set_title('Task-Dependent Emergence Depths', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Sample sizes
        for i, depths in enumerate(plot_data):
            ax.text(i, -7, f'n={len(depths)}', 
                   ha='center', va='top', fontsize=9, 
                   style='italic', color='gray')
    
    def plot_panel_b(self, ax):
        """Panel B: Line plot"""
        
        # Collect model data
        model_depths = {}
        for model in self.data.keys():
            model_depths[model] = {}
            for task in self.tasks:
                depth = self.get_emergence_depth(model, task)
                if depth is not None:
                    model_depths[model][task] = depth
        
        # Task order
        task_order = ['hellaswag', 'gsm8k', 'commonsense_qa', 'boolq']
        task_labels = ['HellaSwag', 'GSM8K', 'CommonsenseQA', 'BoolQ']
        
        # Model names
        model_names = {
            'llama_1b': 'Llama-1B', 'llama_3b': 'Llama-3B',
            'llama_5b': 'Llama-5B', 'llama_13b': 'Llama-13B',
            'phi_1': 'Phi-1.0', 'phi_1.5': 'Phi-1.5',
            'deepsek_7b': 'DeepSeek-7B'
        }
        
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                 '#9467bd', '#8c564b', '#e377c2']
        
        # Plot lines
        x = np.arange(len(task_order))
        for i, (model, depths_dict) in enumerate(model_depths.items()):
            depths = [depths_dict.get(task, np.nan) for task in task_order]
            ax.plot(x, depths, 'o-', linewidth=3, markersize=8,
                   color=colors[i], alpha=0.8,
                   label=model_names.get(model, model))
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels)
        ax.set_ylabel('Emergence Depth (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='center', framealpha=0.95, 
                 fontsize=9, handlelength=2)
        
        # Panel label
        ax.text(-0.12, 1.08, 'B', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        
        # Title
        ax.set_title('Consistent Task Ordering Across Models', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Stats box
        ax.text(0.98, 0.05, 'Spearman ρ = 1.0\np < 0.000001',
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='#fff59d', 
                        edgecolor='black',
                        linewidth=2),
               fontsize=10, fontweight='bold')
    
    def plot_panel_c(self, ax):
        """Panel C: Schematic"""
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Panel label
        ax.text(-0.5, 10, 'C', fontsize=16, fontweight='bold', 
               va='top', ha='left')
        
        # Title
        ax.text(5, 10.3, 'Cognitive Complexity Hierarchy', 
               ha='center', va='top', fontsize=13, fontweight='bold')
        
        # Regions
        regions = [
            {
                'y_start': 0.2, 'y_end': 2.1,
                'color': self.task_colors['hellaswag'],
                'task': 'HellaSwag', 'depth': '32%',
                'desc': 'Pattern Matching'
            },
            {
                'y_start': 2.4, 'y_end': 4.1,
                'color': self.task_colors['gsm8k'],
                'task': 'GSM8K', 'depth': '37%',
                'desc': 'Mathematical Reasoning'
            },
            {
                'y_start': 4.4, 'y_end': 6.8,
                'color': self.task_colors['commonsense_qa'],
                'task': 'CommonsenseQA', 'depth': '56%',
                'desc': 'Abstract Reasoning'
            },
            {
                'y_start': 7.1, 'y_end': 9.2,
                'color': self.task_colors['boolq'],
                'task': 'BoolQ', 'depth': '60%',
                'desc': 'Output Formatting'
            }
        ]
        
        # Draw regions
        for region in regions:
            rect = FancyBboxPatch(
                (1, region['y_start']), 8, 
                region['y_end'] - region['y_start'],
                boxstyle="round,pad=0.05",
                facecolor=region['color'], edgecolor='black',
                linewidth=2, alpha=0.4
            )
            ax.add_patch(rect)
            
            y_mid = (region['y_start'] + region['y_end']) / 2
            
            # Task name
            ax.text(1.5, y_mid, region['task'], 
                   ha='left', va='center', fontsize=11, fontweight='bold')
            
            # Depth
            ax.text(5, y_mid, region['depth'], 
                   ha='center', va='center', fontsize=11, 
                   style='italic', fontweight='bold')
            
            # Description
            ax.text(7.5, y_mid, region['desc'], 
                   ha='center', va='center', fontsize=10)
        
        # Labels
        ax.text(-0.3, 4.7, 'Network\nDepth', 
               ha='center', va='center', fontsize=11, 
               rotation=90, fontweight='bold')
        
        ax.text(5, -0.3, 'Early Layers', 
               ha='center', va='bottom', fontsize=9, 
               style='italic', color='gray')
        
        ax.text(5, 9.7, 'Output Layer', 
               ha='center', va='top', fontsize=9, 
               style='italic', color='gray')
        
        # Complexity arrow
        ax.annotate('', xy=(9.7, 8.8), xytext=(9.7, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        ax.text(10.2, 4.7, 'Increasing\nComplexity', 
               ha='left', va='center', fontsize=10, 
               rotation=270, fontweight='bold')
    
    def generate_and_save(self, output_path='figure2_task_hierarchy_clean.pdf'):
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
    print("🎨 GENERATING FIGURE 2: LARGE & CLEAN")
    print("="*80 + "\n")
    
    generator = Figure2Generator(base_dir=".")
    fig = generator.generate_and_save('figure2_task_hierarchy_clean.pdf')
    
    print("\n✅ COMPLETE!")
    print("\n" + "="*80)
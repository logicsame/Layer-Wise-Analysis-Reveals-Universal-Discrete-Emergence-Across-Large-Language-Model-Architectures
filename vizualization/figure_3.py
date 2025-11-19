"""
Figure 3: Binary Competence Acquisition (FIXED)
Clean Panel C titles with proper spacing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class Figure3Generator:
    """Generate Figure 3: Binary Competence"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.model_dirs = [
            "llama_1b_models", "llama_3b_model", "llama_5b_model",
            "llama_13b_models", "phi_1_models", "phi_1.5_models",
            "deepsek_7b_models"
        ]
        self.tasks = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        self.data = {}
        
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
    
    def extract_model_size(self, model_name):
        """Extract model size in billions"""
        import re
        match = re.search(r'(\d+\.?\d*)b', model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))
        if 'phi_1.5' in model_name or 'phi-1.5' in model_name:
            return 1.5
        if 'phi_1' in model_name or 'phi-1' in model_name:
            return 1.3
        return None
    
    def get_pre_post_accuracy(self, model_name, task):
        """Get pre-emergence and post-emergence accuracy"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None, None
        
        layer_stats = self.data[model_name][task].get('layer_statistics', [])
        cryst_layer = self.data[model_name][task].get('crystallization_layer', -1)
        
        if not layer_stats or cryst_layer < 1:
            return None, None
        
        pre_idx = cryst_layer - 1
        if pre_idx < 0 or pre_idx >= len(layer_stats):
            return None, None
        pre_acc = layer_stats[pre_idx]['accuracy']
        post_acc = layer_stats[-1]['accuracy']
        
        return pre_acc * 100, post_acc * 100
    
    def get_final_accuracy(self, model_name, task):
        """Get final accuracy"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None
        return self.data[model_name][task].get('overall_accuracy', None)
    
    def get_layer_accuracies(self, model_name, task):
        """Get full layer-by-layer accuracies"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None, None
        
        layer_stats = self.data[model_name][task].get('layer_statistics', [])
        if not layer_stats:
            return None, None
        
        layers = [s['layer'] for s in layer_stats]
        accs = [s['accuracy'] * 100 for s in layer_stats]
        
        return np.array(layers), np.array(accs)
    
    def create_figure(self):
        """Create 4-panel figure"""
        
        # Large figure
        fig = plt.figure(figsize=(16, 12))
        
        # GridSpec with more space for Panel C
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                             left=0.08, right=0.96, 
                             top=0.93, bottom=0.07)
        
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        
        # Panel C: 2x2 subgrid with MORE spacing
        gs_c = gs[1, 0].subgridspec(2, 2, hspace=0.45, wspace=0.35)
        ax_c = [fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(2)]
        
        ax_d = fig.add_subplot(gs[1, 1])
        
        self.plot_panel_a(ax_a)
        self.plot_panel_b(ax_b)
        self.plot_panel_c(ax_c)
        self.plot_panel_d(ax_d)
        
        return fig
    
    def plot_panel_a(self, ax):
        """Panel A: Pre vs post scatter"""
        
        pre_accs = []
        post_accs = []
        
        for model in self.data.keys():
            for task in self.tasks:
                pre, post = self.get_pre_post_accuracy(model, task)
                if pre is not None and post is not None:
                    pre_accs.append(pre)
                    post_accs.append(post)
        
        pre_accs = np.array(pre_accs)
        post_accs = np.array(post_accs)
        
        # Diagonal line
        ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.5,
               label='Gradual Learning\nPrediction', zorder=1)
        
        # Scatter
        ax.scatter(pre_accs, post_accs, s=150, alpha=0.7,
                  c='#e74c3c', edgecolors='black', linewidth=1.5,
                  zorder=3, label=f'Observed (n={len(pre_accs)})')
        
        ax.set_xlabel('Pre-Emergence Accuracy (%)\n(Layer Before Crystallization)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Post-Emergence Accuracy (%)\n(Final Layer)', 
                     fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        ax.set_aspect('equal')
        
        ax.text(-0.15, 1.08, 'A', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Binary Competence: Pre vs Post-Emergence', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        
        mean_pre = np.mean(pre_accs)
        mean_post = np.mean(post_accs)
        ax.text(0.05, 0.95, 
               f'Mean jump:\n{mean_pre:.1f}% → {mean_post:.1f}%',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=2),
               fontsize=11, fontweight='bold')
    
    def plot_panel_b(self, ax):
        """Panel B: Final accuracy distribution"""
        
        final_accs = []
        for model in self.data.keys():
            for task in self.tasks:
                acc = self.get_final_accuracy(model, task)
                if acc is not None:
                    final_accs.append(acc * 100)
        
        final_accs = np.array(final_accs)
        
        bins = np.arange(98, 101, 0.2)
        ax.hist(final_accs, bins=bins, color='#3498db', alpha=0.7,
               edgecolor='black', linewidth=1.5)
        
        mean_acc = np.mean(final_accs)
        std_acc = np.std(final_accs)
        cv = (std_acc / mean_acc) * 100
        
        ax.axvline(mean_acc, color='red', linewidth=3, linestyle='-',
                  label=f'Mean: {mean_acc:.2f}%')
        ax.axvline(mean_acc - std_acc, color='orange', linewidth=2, 
                  linestyle='--', label=f'±1 SD: {std_acc:.2f}%')
        ax.axvline(mean_acc + std_acc, color='orange', linewidth=2, 
                  linestyle='--')
        
        ax.set_xlabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Model-Task Pairs', fontsize=12, fontweight='bold')
        ax.set_xlim(98.5, 100.5)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
        
        ax.text(-0.15, 1.08, 'B', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Convergence to Maximal Performance', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        ax.text(0.95, 0.95, 
               f'CV = {cv:.2f}%\n(Extremely Low Variance)',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='#fff59d', 
                        edgecolor='black', linewidth=2),
               fontsize=11, fontweight='bold')
    
    def plot_panel_c(self, axes):
        """Panel C: Example emergence profiles (4 subplots) - FIXED TITLES"""
        
        # Examples with CLEAN titles
        examples = [
            ('llama_1b', 'gsm8k', 'Llama-1B (GSM8K)'),
            ('llama_5b', 'hellaswag', 'Llama-5B (HellaSwag)'),
            ('phi_1', 'boolq', 'Phi-1.0 (BoolQ)'),
            ('deepsek_7b', 'commonsense_qa', 'DeepSeek-7B (CommonsenseQA)')
        ]
        
        for idx, (ax, (model, task, label)) in enumerate(zip(axes, examples)):
            layers, accs = self.get_layer_accuracies(model, task)
            
            if layers is None:
                continue
            
            # Plot curve
            ax.plot(layers, accs, 'o-', linewidth=3, markersize=6,
                   color='#2ecc71', alpha=0.8)
            
            # Crystallization point
            cryst_layer = self.data[model][task].get('crystallization_layer', -1)
            if cryst_layer >= 0:
                cryst_idx = np.argmin(np.abs(layers - cryst_layer))
                cryst_acc = accs[cryst_idx]
                ax.axvline(cryst_layer, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Crystallization')
                ax.scatter([cryst_layer], [cryst_acc], s=200, 
                          color='red', edgecolors='black', 
                          linewidth=2, zorder=5, marker='*')
            
            # Formatting
            ax.set_xlabel('Layer Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
            
            # TITLE - single line, centered
            ax.set_title(label, fontsize=11, fontweight='bold', pad=8)
            
            if idx == 0:
                ax.legend(loc='lower right', fontsize=9)
        
        # Panel label C (on first subplot, positioned higher)
        axes[0].text(-0.28, 1.20, 'C', transform=axes[0].transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
    
    def plot_panel_d(self, ax):
        """Panel D: Size independence"""
        
        sizes = []
        accs = []
        
        for model in self.data.keys():
            size = self.extract_model_size(model)
            if size is None:
                continue
            
            model_accs = []
            for task in self.tasks:
                acc = self.get_final_accuracy(model, task)
                if acc is not None:
                    model_accs.append(acc * 100)
            
            if model_accs:
                sizes.append(size)
                accs.append(np.mean(model_accs))
        
        sizes = np.array(sizes)
        accs = np.array(accs)
        
        ax.scatter(sizes, accs, s=200, alpha=0.7,
                  c='#9b59b6', edgecolors='black', linewidth=2)
        
        mean_acc = np.mean(accs)
        ax.axhline(mean_acc, color='red', linewidth=3, linestyle='-',
                  label=f'Mean: {mean_acc:.2f}%')
        
        std_acc = np.std(accs)
        ax.fill_between([0.8, 15], mean_acc - std_acc, mean_acc + std_acc,
                       color='red', alpha=0.2, label=f'±1 SD: {std_acc:.2f}%')
        
        ax.set_xlabel('Model Size (Billions of Parameters)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Final Accuracy (%)', 
                     fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(0.8, 15)
        ax.set_ylim(98.5, 100.5)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        
        ax.text(-0.15, 1.08, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
        ax.set_title('Size-Independent Convergence', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        
        from scipy.stats import pearsonr
        r, p = pearsonr(np.log10(sizes), accs)
        ax.text(0.05, 0.05, 
               f'Pearson r = {r:.3f}\np = {p:.3f}\n(No size dependence)',
               transform=ax.transAxes, ha='left', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=2),
               fontsize=10, fontweight='bold')
    
    def generate_and_save(self, output_path='figure3_binary_competence_fixed.pdf'):
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
    print("🎨 GENERATING FIGURE 3: BINARY COMPETENCE (FIXED)")
    print("="*80 + "\n")
    
    generator = Figure3Generator(base_dir=".")
    fig = generator.generate_and_save('figure3_binary_competence_fixed.pdf')
    
    print("\n✅ COMPLETE - CLEAN TITLES!")
    print("\nFixed:")
    print("  ✅ Panel C titles: single line, no overlap")
    print("  ✅ Increased vertical spacing (hspace=0.45)")
    print("  ✅ Clean labels: 'Llama-1B (GSM8K)' etc.")
    print("\n" + "="*80)
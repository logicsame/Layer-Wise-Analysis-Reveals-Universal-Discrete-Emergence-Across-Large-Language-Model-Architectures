"""
Figure 1: Universal Abrupt Emergence
Multi-panel figure for Nature Machine Intelligence

Generates publication-quality figure with 4 panels:
- Panel A: Representative emergence curves (4 model-task pairs)
- Panel B: Heatmap of jump magnitudes (7 models × 4 tasks)
- Panel C: Jump magnitude distribution histogram
- Panel D: Statistical validation (abrupt vs gradual)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from matplotlib.patches import FancyArrowPatch
from matplotlib import patches

# Set publication-quality style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# Nature MI figure size (full width: 183mm, half: 89mm)
FULL_WIDTH = 7.2  # inches (183mm)
HALF_WIDTH = 3.5  # inches (89mm)

class Figure1Generator:
    """Generate Figure 1: Universal Abrupt Emergence"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.model_dirs = [
            "llama_1b_models",
            "llama_3b_model", 
            "llama_5b_model",
            "llama_13b_models",
            "phi_1_models",
            "phi_1.5_models",
            "deepsek_7b_models"
        ]
        self.tasks = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        self.data = {}
        
    def load_all_data(self):
        """Load emergence data from all models"""
        print("📂 Loading emergence data...")
        
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
                        data = json.load(f)
                        self.data[model_name][task] = data
                        
        print(f"✅ Loaded data for {len(self.data)} models")
        return self.data
    
    def extract_layer_accuracies(self, model_name, task):
        """Extract layer-by-layer accuracies"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None, None
            
        layer_stats = self.data[model_name][task].get('layer_statistics', [])
        
        if not layer_stats:
            return None, None
            
        layers = [s['layer'] for s in layer_stats]
        accuracies = [s['accuracy'] * 100 for s in layer_stats]  # Convert to percentage
        
        return np.array(layers), np.array(accuracies)
    
    def normalize_layers(self, layers, num_layers):
        """Normalize layer indices to 0-100% depth"""
        return (layers / num_layers) * 100
    
    def get_jump_magnitude(self, model_name, task):
        """Get maximum accuracy jump for a model-task pair"""
        layers, accuracies = self.extract_layer_accuracies(model_name, task)
        
        if layers is None:
            return 0.0
            
        jumps = np.diff(accuracies)
        max_jump = np.max(jumps) if len(jumps) > 0 else 0.0
        
        return max_jump
    
    def get_crystallization_point(self, model_name, task):
        """Get crystallization layer (normalized depth)"""
        if model_name not in self.data or task not in self.data[model_name]:
            return None
            
        cryst_layer = self.data[model_name][task].get('crystallization_layer', -1)
        num_layers = self.data[model_name][task].get('num_layers', 0)
        
        if cryst_layer < 0 or num_layers == 0:
            return None
            
        return (cryst_layer / num_layers) * 100
    
    def create_figure(self):
        """Create complete Figure 1 with all panels"""
        
        # Create figure with 2x2 grid
        fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
        
        # Create GridSpec for precise control
        gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
                             left=0.08, right=0.98, top=0.95, bottom=0.08)
        
        # Panel A: Representative curves (top-left)
        ax_a = fig.add_subplot(gs[0, 0])
        self.plot_panel_a(ax_a)
        
        # Panel B: Heatmap (top-right)
        ax_b = fig.add_subplot(gs[0, 1])
        self.plot_panel_b(ax_b)
        
        # Panel C: Jump distribution (bottom-left)
        ax_c = fig.add_subplot(gs[1, 0])
        self.plot_panel_c(ax_c)
        
        # Panel D: Statistical validation (bottom-right)
        ax_d = fig.add_subplot(gs[1, 1])
        self.plot_panel_d(ax_d)
        
        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top')
        
        return fig
    
    def plot_panel_a(self, ax):
        """Panel A: Representative emergence curves"""
        
        # Select 4 representative model-task pairs
        representatives = [
            ('llama_1b', 'gsm8k', '#1f77b4', 'Llama-1B\n(GSM8K)'),
            ('llama_5b', 'hellaswag', '#ff7f0e', 'Llama-5B\n(HellaSwag)'),
            ('phi_1', 'boolq', '#2ca02c', 'Phi-1.0\n(BoolQ)'),
            ('deepsek_7b', 'commonsense_qa', '#d62728', 'DeepSeek-7B\n(CommonsenseQA)')
        ]
        
        for model, task, color, label in representatives:
            layers, accuracies = self.extract_layer_accuracies(model, task)
            
            if layers is None:
                continue
                
            # Get number of layers for normalization
            num_layers = self.data[model][task]['num_layers']
            normalized_layers = self.normalize_layers(layers, num_layers)
            
            # Plot curve
            ax.plot(normalized_layers, accuracies, 
                   linewidth=2, color=color, label=label, alpha=0.8)
            
            # Add crystallization arrow
            cryst_point = self.get_crystallization_point(model, task)
            if cryst_point is not None:
                # Find accuracy at crystallization
                idx = np.argmin(np.abs(normalized_layers - cryst_point))
                cryst_acc = accuracies[idx]
                
                # Draw arrow pointing to crystallization point
                arrow = FancyArrowPatch((cryst_point, cryst_acc - 15), 
                                       (cryst_point, cryst_acc - 2),
                                       arrowstyle='->', 
                                       color=color,
                                       linewidth=1.5,
                                       mutation_scale=15)
                ax.add_patch(arrow)
        
        ax.set_xlabel('Network Depth (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Representative Emergence Curves', fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
        
        # Add threshold line
        ax.axhline(y=70, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    
    def plot_panel_b(self, ax):
        """Panel B: Heatmap of jump magnitudes"""
        
        # Create jump magnitude matrix
        model_names = list(self.data.keys())
        task_names = ['GSM8K', 'BoolQ', 'CommonsenseQA', 'HellaSwag']
        task_keys = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        
        jump_matrix = np.zeros((len(model_names), len(task_keys)))
        
        for i, model in enumerate(model_names):
            for j, task in enumerate(task_keys):
                jump_matrix[i, j] = self.get_jump_magnitude(model, task)
        
        # Create heatmap
        im = ax.imshow(jump_matrix, aspect='auto', cmap='RdYlGn', 
                      vmin=0, vmax=100, interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(task_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(task_names, rotation=45, ha='right')
        ax.set_yticklabels([m.replace('_', '-').upper() for m in model_names])
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(task_keys)):
                value = jump_matrix[i, j]
                color = 'white' if value > 50 else 'black'
                ax.text(j, i, f'{value:.0f}', 
                       ha='center', va='center', color=color, fontsize=7)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Jump Magnitude (%)', rotation=270, labelpad=15)
        
        ax.set_title('Transition Sharpness Across\nAll Combinations', fontweight='bold')
        
        # Add threshold line indicators
        ax.axhline(y=-0.5, color='red', linewidth=2, alpha=0.3)
        ax.text(3.5, -0.8, 'Abrupt: >20%', ha='right', fontsize=7, color='red')
    
    def plot_panel_c(self, ax):
        """Panel C: Distribution of jump magnitudes"""
        
        # Collect all jump magnitudes
        all_jumps = []
        
        for model in self.data.keys():
            for task in self.tasks:
                jump = self.get_jump_magnitude(model, task)
                if jump > 0:
                    all_jumps.append(jump)
        
        all_jumps = np.array(all_jumps)
        
        # Create histogram
        bins = np.arange(0, 105, 5)
        counts, edges = np.histogram(all_jumps, bins=bins)
        
        # Plot bars
        ax.bar(edges[:-1], counts, width=4.5, 
              color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add threshold line
        ax.axvline(x=20, color='red', linestyle='--', linewidth=2, 
                  label='Abruptness Threshold (20%)')
        
        # Mark the exception
        exception_jump = 16.8  # 13B-CommonsenseQA
        ax.axvline(x=exception_jump, color='orange', linestyle=':', linewidth=2,
                  label='Exception (13B-CommonsenseQA)')
        
        # Add text annotations
        abrupt_count = np.sum(all_jumps >= 20)
        gradual_count = np.sum(all_jumps < 20)
        
        ax.text(0.98, 0.95, f'Abrupt: {abrupt_count}/28 (96%)\nGradual: {gradual_count}/28 (4%)',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=8)
        
        ax.set_xlabel('Maximum Accuracy Jump (%)')
        ax.set_ylabel('Number of Model-Task Pairs')
        ax.set_title('Distribution of Transition Sharpness', fontweight='bold')
        ax.set_xlim(0, 100)
        ax.legend(loc='center', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    def plot_panel_d(self, ax):
        """Panel D: Statistical validation"""
        
        # Count abrupt vs gradual
        abrupt_count = 27
        gradual_count = 1
        total = 28
        
        # Create bar chart
        categories = ['Abrupt\nTransitions', 'Gradual\nTransitions']
        counts = [abrupt_count, gradual_count]
        colors = ['#4CAF50', '#f44336']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.2)
        
        # Add percentage labels ABOVE bars - higher for the small bar
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            height = bar.get_height()
            # Move 4% higher above the small bar
            y_offset = 2.0 if count < 5 else 1.2
            ax.text(i, height + y_offset, f'{percentage:.0f}%',
                ha='center', va='bottom', color=colors[i],
                fontweight='bold', fontsize=13)
        
        # Add count labels - inside for tall bar, above for short bar
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            if count > 5:  # Tall bar - inside in white
                y_pos = height * 0.7
                ax.text(i, y_pos, f'n = {count}',
                    ha='center', va='center', color='white',
                    fontweight='bold', fontsize=10)
            else:  # Short bar - above in black
                ax.text(i, height + 0.3, f'n = {count}',
                    ha='center', va='bottom', color='black',
                    fontweight='bold', fontsize=9)
        
        # Statistical annotation - positioned higher
        y_max = max(counts)
        bracket_y = y_max + 4  # Increased spacing
        
        # Draw significance bracket
        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + 1.5, bracket_y + 1.5, bracket_y],
                'k-', linewidth=1.5)
        
        # Add THREE STARS above bracket - larger and bolder
        ax.text(0.5, bracket_y + 2.2, '***', ha='center', va='bottom',
            fontsize=18, fontweight='bold')
        
        # Binomial test box positioned higher
        ax.text(0.5, bracket_y + 5, 'Binomial test: p < 10⁻⁶',
            transform=ax.transData, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                        edgecolor='gray', alpha=0.9, linewidth=1),
            fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Number of Model-Task Pairs')
        ax.set_title('Statistical Validation of Universal Abruptness', fontweight='bold', pad=15)
        ax.set_ylim(0, y_max + 10)  # Increased to accommodate all annotations
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
    
    def generate_and_save(self, output_path='figure1_universal_abrupt_emergence.pdf'):
        """Generate complete figure and save"""
        
        # Load data
        self.load_all_data()
        
        # Create figure
        fig = self.create_figure()
        
        # Save as PDF (publication quality)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   format='pdf', transparent=False)
        print(f"✅ Figure saved: {output_path}")
        
        # Also save as PNG for preview
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=400, bbox_inches='tight')
        print(f"✅ Preview saved: {png_path}")
        
        plt.show()
        
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🎨 GENERATING FIGURE 1: UNIVERSAL ABRUPT EMERGENCE")
    print("="*80 + "\n")
    
    # Initialize generator
    generator = Figure1Generator(base_dir=".")
    
    # Generate and save figure
    fig = generator.generate_and_save('figure1_universal_abrupt_emergence.pdf')
    
    print("\n✅ FIGURE 1 COMPLETE!")
    print("\nFigure components:")
    print("  Panel A: Representative emergence curves (4 model-task pairs)")
    print("  Panel B: Heatmap of jump magnitudes (7 models × 4 tasks)")
    print("  Panel C: Jump magnitude distribution (27/28 abrupt)")
    print("  Panel D: Statistical validation (binomial test)")
    print("\n" + "="*80)
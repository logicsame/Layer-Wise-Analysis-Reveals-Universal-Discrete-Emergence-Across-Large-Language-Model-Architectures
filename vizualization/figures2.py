"""
Generate Supplementary Figure S3: Complete Layer-wise Accuracy Trajectories
Shows all 28 model-task combinations with transition characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7

class CompleteTrajectoryVisualizer:
    """Visualize all 28 model-task accuracy trajectories"""
    
    def __init__(self, base_dir="E:/publication/ed/viz"):
        self.base_dir = Path(base_dir)
        
        self.model_configs = [
            {'dir': 'llama_1b_models', 'name': 'Llama-1B', 'family': 'Llama'},
            {'dir': 'llama_3b_model', 'name': 'Llama-3B', 'family': 'Llama'},
            {'dir': 'llama_5b_model', 'name': 'Llama-5B', 'family': 'Llama'},
            {'dir': 'llama_13b_models', 'name': 'Llama-13B', 'family': 'Llama'},
            {'dir': 'phi_1_models', 'name': 'Phi-1.0', 'family': 'Phi'},
            {'dir': 'phi_1.5_models', 'name': 'Phi-1.5', 'family': 'Phi'},
            {'dir': 'deepsek_7b_models', 'name': 'DeepSeek-7B', 'family': 'DeepSeek'},
        ]
        
        self.tasks = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        self.task_labels = {
            'gsm8k': 'GSM8K\n(Math)',
            'boolq': 'BoolQ\n(Boolean)',
            'commonsense_qa': 'CommonsenseQA\n(Commonsense)',
            'hellaswag': 'HellaSwag\n(Language)'
        }
        
        self.family_colors = {
            'Llama': '#3498db',
            'Phi': '#e74c3c',
            'DeepSeek': '#2ecc71'
        }
    
    def logistic_function(self, x, A, k, x0):
        """Logistic function for fitting"""
        return A / (1 + np.exp(-k * (x - x0)))
    
    def load_trajectory(self, model_dir, task_name):
        """Load layer-wise accuracy trajectory"""
        probing_files = list((model_dir / "probing").glob("*_combined.csv"))
        
        if not probing_files:
            return None
        
        df = pd.read_csv(probing_files[0])
        df_task = df[df['dataset_name'] == task_name]
        
        if len(df_task) == 0:
            return None
        
        # Calculate layer-wise accuracy
        layer_acc = df_task.groupby('layer_idx')['is_correct'].mean()
        num_layers = int(df_task['layer_idx'].max() + 1)
        
        # Ensure all layers present
        layers = np.arange(num_layers)
        accuracies = np.array([layer_acc.get(i, 0.0) for i in layers])
        
        return layers, accuracies
    
    def fit_logistic(self, layers, accuracies):
        """Fit logistic model and return parameters"""
        valid_mask = ~np.isnan(accuracies)
        if valid_mask.sum() < 5:
            return None, None
        
        layers_valid = layers[valid_mask]
        acc_valid = accuracies[valid_mask]
        
        A_init = np.max(acc_valid)
        x0_init = np.median(layers_valid[acc_valid > 0.5 * A_init]) if any(acc_valid > 0.5 * A_init) else len(layers_valid) / 2
        k_init = 1.0
        
        try:
            popt, _ = curve_fit(
                self.logistic_function,
                layers_valid,
                acc_valid,
                p0=[A_init, k_init, x0_init],
                maxfev=5000,
                bounds=([0, 0, 0], [1.5, 10, len(layers)])
            )
            return popt, layers_valid
        except:
            return None, None
    
    def create_figure(self):
        """
        Create 7x4 grid: 7 models (rows) × 4 tasks (columns)
        """
        fig = plt.figure(figsize=(16, 18))
        
        gs = fig.add_gridspec(7, 4, hspace=0.35, wspace=0.3,
                             left=0.06, right=0.98,
                             top=0.96, bottom=0.04)
        
        # Statistics for summary
        all_stats = []
        
        for row, model_config in enumerate(self.model_configs):
            model_dir = self.base_dir / model_config['dir']
            model_name = model_config['name']
            family = model_config['family']
            color = self.family_colors[family]
            
            for col, task_name in enumerate(self.tasks):
                ax = fig.add_subplot(gs[row, col])
                
                # Load trajectory
                trajectory = self.load_trajectory(model_dir, task_name)
                
                if trajectory is None:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                layers, accuracies = trajectory
                acc_pct = accuracies * 100
                
                # Plot trajectory
                ax.plot(layers, acc_pct, 'o-', linewidth=2, markersize=4,
                       color=color, alpha=0.8, zorder=3)
                
                # Fit logistic
                popt, layers_valid = self.fit_logistic(layers, accuracies)
                
                if popt is not None:
                    # Plot fit
                    x_fit = np.linspace(0, len(layers)-1, 100)
                    y_fit = self.logistic_function(x_fit, *popt) * 100
                    ax.plot(x_fit, y_fit, '--', linewidth=1.5,
                           color='gray', alpha=0.5, zorder=2)
                    
                    k_value = popt[1]
                    x0_value = popt[2]
                    
                    # Mark inflection point
                    ax.axvline(x0_value, color='red', linestyle=':',
                              linewidth=1, alpha=0.5)
                    
                    # Store stats
                    all_stats.append({
                        'model': model_name,
                        'task': task_name,
                        'k': k_value,
                        'inflection': x0_value,
                        'num_layers': len(layers)
                    })
                    
                    # Annotate k value
                    ax.text(0.95, 0.05, f'k={k_value:.2f}',
                           transform=ax.transAxes, ha='right', va='bottom',
                           fontsize=7, bbox=dict(boxstyle='round',
                                                facecolor='white',
                                                alpha=0.7))
                
                # Mark 70% threshold
                ax.axhline(70, color='green', linestyle='--',
                          linewidth=1, alpha=0.3)
                
                # Formatting
                ax.set_ylim(-5, 105)
                ax.set_xlim(-0.5, len(layers)-0.5)
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                
                # Labels
                if col == 0:
                    ax.set_ylabel(f'{model_name}\nAccuracy (%)',
                                fontsize=9, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
                if row == 0:
                    ax.set_title(self.task_labels[task_name],
                               fontsize=10, fontweight='bold')
                
                if row == len(self.model_configs) - 1:
                    ax.set_xlabel('Layer Index', fontsize=9)
                else:
                    ax.set_xlabel('')
                
                # Reduce tick density
                ax.locator_params(axis='x', nbins=5)
                ax.locator_params(axis='y', nbins=5)
        
        # Add overall title
        fig.suptitle('Complete Layer-wise Accuracy Trajectories: All 28 Model-Task Combinations',
                    fontsize=14, fontweight='bold', y=0.99)
        
        return fig, all_stats
    
    def create_summary_table(self, all_stats):
        """Create summary statistics table"""
        if not all_stats:
            return None
        
        df_stats = pd.DataFrame(all_stats)
        
        # Overall statistics
        print("\n" + "="*80)
        print("📊 TRAJECTORY STATISTICS SUMMARY")
        print("="*80)
        
        print("\n🔹 By Task:")
        task_summary = df_stats.groupby('task').agg({
            'k': ['mean', 'std', 'min', 'max'],
            'inflection': ['mean', 'std']
        }).round(2)
        print(task_summary)
        
        print("\n🔹 By Model:")
        model_summary = df_stats.groupby('model').agg({
            'k': ['mean', 'std'],
            'inflection': ['mean', 'std']
        }).round(2)
        print(model_summary)
        
        return df_stats
    
    def generate_and_save(self, output_path='figureS3_all_trajectories.pdf'):
        """Generate and save figure"""
        fig, stats = self.create_figure()
        
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   format='pdf', transparent=False)
        print(f"\n✅ PDF: {output_path}")
        
        png_path = output_path.with_suffix('.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ PNG: {png_path}")
        
        # Create summary
        df_stats = self.create_summary_table(stats)
        
        if df_stats is not None:
            stats_path = output_path.parent / "figureS3_statistics.csv"
            df_stats.to_csv(stats_path, index=False)
            print(f"✅ Stats CSV: {stats_path}")
        
        plt.show()
        return fig, df_stats


# ============================================================================
# MAIN FOR FIGURE S3
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🎨 GENERATING FIGURE S3: ALL ACCURACY TRAJECTORIES")
    print("="*80 + "\n")
    
    visualizer = CompleteTrajectoryVisualizer(base_dir="E:/publication/ed/viz")
    fig, stats = visualizer.generate_and_save('E:/publication/ed/figureS3_all_trajectories.pdf')
    
    print("\n✅ FIGURE S3 COMPLETE!")
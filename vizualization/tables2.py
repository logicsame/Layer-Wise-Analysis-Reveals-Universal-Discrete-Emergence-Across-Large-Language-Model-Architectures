"""
Generate Supplementary Table S2: Detection Robustness Analysis
Tests sensitivity to parameter variations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class RobustnessAnalyzer:
    """Analyze robustness of emergence detection to parameter variations"""
    
    def __init__(self, base_dir="E:/publication/ed/viz"):
        self.base_dir = Path(base_dir)
        
        self.model_configs = [
            {'dir': 'llama_1b_models', 'name': 'Llama-1B'},
            {'dir': 'llama_3b_model', 'name': 'Llama-3B'},
            {'dir': 'llama_5b_model', 'name': 'Llama-5B'},
            {'dir': 'llama_13b_models', 'name': 'Llama-13B'},
            {'dir': 'phi_1_models', 'name': 'Phi-1.0'},
            {'dir': 'phi_1.5_models', 'name': 'Phi-1.5'},
            {'dir': 'deepsek_7b_models', 'name': 'DeepSeek-7B'},
        ]
        
        self.tasks = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        
        # Baseline parameters
        self.baseline_params = {
            'accuracy_threshold': 0.7,
            'stability_window': 3,
            'stability_epsilon': 0.05,
            'confidence_threshold': 0.01
        }
    
    def find_crystallization_layer(self, layer_stats, params):
        """
        Find crystallization layer with given parameters
        """
        acc_thresh = params['accuracy_threshold']
        stab_window = params['stability_window']
        stab_eps = params['stability_epsilon']
        conf_thresh = params['confidence_threshold']
        
        num_layers = len(layer_stats)
        skip_layers = max(3, num_layers // 10)
        
        for layer_idx in range(skip_layers, num_layers - stab_window + 1):
            if layer_stats[layer_idx]['accuracy'] < acc_thresh:
                continue
            
            window_accs = [layer_stats[i]['accuracy'] 
                          for i in range(layer_idx, min(layer_idx + stab_window, num_layers))]
            
            if all(acc >= acc_thresh - stab_eps for acc in window_accs):
                if layer_stats[layer_idx]['confidence'] >= conf_thresh:
                    return layer_idx
        
        # Fallback
        valid_layers = range(skip_layers, num_layers)
        if valid_layers:
            return max(valid_layers, key=lambda i: layer_stats[i]['accuracy'])
        
        return num_layers - 1
    
    def load_layer_stats(self, model_dir, task_name):
        """Load layer statistics for a model-task combination"""
        probing_files = list((model_dir / "probing").glob("*_combined.csv"))
        
        if not probing_files:
            return None
        
        df = pd.read_csv(probing_files[0])
        df_task = df[df['dataset_name'] == task_name]
        
        if len(df_task) == 0:
            return None
        
        num_layers = int(df_task['layer_idx'].max() + 1)
        
        layer_stats = []
        for layer_idx in range(num_layers):
            layer_data = df_task[df_task['layer_idx'] == layer_idx]
            if len(layer_data) == 0:
                layer_stats.append({'layer': layer_idx, 'accuracy': 0.0, 'confidence': 0.0})
            else:
                layer_stats.append({
                    'layer': layer_idx,
                    'accuracy': float(layer_data['is_correct'].mean()),
                    'confidence': float(layer_data['layer_confidence'].mean())
                })
        
        return layer_stats
    
    def run_ablation_study(self):
        """
        Run complete ablation study with parameter variations
        """
        print("\n" + "="*80)
        print("🔬 DETECTION ROBUSTNESS ABLATION STUDY")
        print("="*80 + "\n")
        
        # Define parameter variations
        variations = [
            {'name': 'Baseline', 'params': self.baseline_params},
            {'name': 'Stricter Accuracy (0.8)', 'params': {**self.baseline_params, 'accuracy_threshold': 0.8}},
            {'name': 'Looser Accuracy (0.6)', 'params': {**self.baseline_params, 'accuracy_threshold': 0.6}},
            {'name': 'Longer Stability (5)', 'params': {**self.baseline_params, 'stability_window': 5}},
            {'name': 'Shorter Stability (1)', 'params': {**self.baseline_params, 'stability_window': 1}},
            {'name': 'Higher Confidence (0.05)', 'params': {**self.baseline_params, 'confidence_threshold': 0.05}},
            {'name': 'Lower Confidence (0.001)', 'params': {**self.baseline_params, 'confidence_threshold': 0.001}},
        ]
        
        # Store results for each variation
        all_results = {}
        
        for var in variations:
            print(f"🔍 Testing: {var['name']}")
            results = []
            
            for model_config in self.model_configs:
                model_dir = self.base_dir / model_config['dir']
                model_name = model_config['name']
                
                for task_name in self.tasks:
                    layer_stats = self.load_layer_stats(model_dir, task_name)
                    
                    if layer_stats is None:
                        continue
                    
                    cryst_layer = self.find_crystallization_layer(layer_stats, var['params'])
                    
                    results.append({
                        'model': model_name,
                        'task': task_name,
                        'crystallization_layer': cryst_layer,
                        'num_layers': len(layer_stats)
                    })
            
            all_results[var['name']] = pd.DataFrame(results)
            print(f"   ✅ Detected {len(results)} combinations")
        
        # Analyze results
        analysis = self.analyze_variations(all_results)
        
        return analysis, all_results
    
    def analyze_variations(self, all_results):
        """
        Analyze agreement and shifts across parameter variations
        """
        baseline = all_results['Baseline']
        
        analysis = []
        
        for var_name, var_df in all_results.items():
            if var_name == 'Baseline':
                analysis.append({
                    'variation': 'Baseline',
                    'mean_shift': 0.0,
                    'agreement_pct': 100.0,
                    'max_shift': 0,
                    'task_hierarchy_rho': 1.0
                })
                continue
            
            # Merge with baseline
            merged = baseline.merge(var_df, on=['model', 'task'], suffixes=('_base', '_var'))
            
            # Calculate shifts
            merged['shift'] = merged['crystallization_layer_var'] - merged['crystallization_layer_base']
            
            mean_shift = merged['shift'].mean()
            agreement = (merged['shift'] == 0).mean() * 100
            max_shift = merged['shift'].abs().max()
            
            # Task hierarchy preservation
            task_ranks_base = baseline.groupby('task')['crystallization_layer'].mean().rank()
            task_ranks_var = var_df.groupby('task')['crystallization_layer'].mean().rank()
            
            common_tasks = set(task_ranks_base.index) & set(task_ranks_var.index)
            if len(common_tasks) >= 2:
                rho, _ = spearmanr(
                    [task_ranks_base[t] for t in common_tasks],
                    [task_ranks_var[t] for t in common_tasks]
                )
            else:
                rho = np.nan
            
            analysis.append({
                'variation': var_name,
                'mean_shift': mean_shift,
                'agreement_pct': agreement,
                'max_shift': int(max_shift),
                'task_hierarchy_rho': rho
            })
        
        return pd.DataFrame(analysis)
    
    def format_latex_table(self, analysis_df):
        """Format results as LaTeX table"""
        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\caption{\\textbf{Detection robustness analysis across parameter variations.}}")
        latex.append("\\label{tab:robustness}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Parameter Variation} & \\textbf{Mean Shift} & \\textbf{Agreement} & \\textbf{Max Shift} & \\textbf{Hierarchy} \\\\")
        latex.append(" & \\textbf{(layers)} & \\textbf{(\\%)} & \\textbf{(layers)} & \\textbf{$\\rho$} \\\\")
        latex.append("\\midrule")
        
        for _, row in analysis_df.iterrows():
            var_name = row['variation']
            mean_shift = f"{row['mean_shift']:+.1f}" if row['mean_shift'] != 0 else "0.0"
            agreement = f"{row['agreement_pct']:.0f}"
            max_shift = f"{row['max_shift']:.0f}"
            rho = f"{row['task_hierarchy_rho']:.3f}" if not np.isnan(row['task_hierarchy_rho']) else "---"
            
            latex.append(f"{var_name} & {mean_shift} & {agreement} & {max_shift} & {rho} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_table(self, output_path='table_s2_robustness.tex'):
        """Generate and save table"""
        analysis, all_results = self.run_ablation_study()
        
        print("\n" + "="*80)
        print("📊 ROBUSTNESS ANALYSIS RESULTS")
        print("="*80)
        print(analysis.to_string(index=False))
        
        # Save LaTeX
        latex_content = self.format_latex_table(analysis)
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        print(f"\n💾 LaTeX table: {output_path}")
        
        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        analysis.to_csv(csv_path, index=False)
        print(f"💾 CSV data: {csv_path}")
        
        return analysis, all_results


# ============================================================================
# MAIN FOR TABLE S2
# ============================================================================

if __name__ == "__main__":
    
    analyzer = RobustnessAnalyzer(base_dir="E:/publication/ed/viz")
    analysis, results = analyzer.generate_table('E:/publication/ed/table_s2_robustness.tex')
    
    print("\n✅ TABLE S2 COMPLETE!")
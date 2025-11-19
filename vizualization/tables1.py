"""
Generate Supplementary Table S1: Comprehensive Per-Model Emergence Statistics
UPDATED to use *_probing_combined.csv files with ALL tasks
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class SupplementaryTableGenerator:
    """Generate comprehensive statistics table for all model-task combinations"""
    
    def __init__(self, base_dir="E:/publication/ed/viz"):
        self.base_dir = Path(base_dir)
        print(f"🔍 Base directory: {self.base_dir}")
        print(f"   Exists: {self.base_dir.exists()}")
        
    def logistic_function(self, x, A, k, x0):
        """Logistic function for fitting transitions"""
        return A / (1 + np.exp(-k * (x - x0)))
    
    def calculate_abruptness(self, layer_accuracies):
        """Calculate transition abruptness (k parameter from logistic fit)"""
        layers = np.arange(len(layer_accuracies))
        accuracies = np.array(layer_accuracies)
        
        valid_mask = ~np.isnan(accuracies)
        if valid_mask.sum() < 5:
            return np.nan, np.nan
        
        layers_valid = layers[valid_mask]
        accuracies_valid = accuracies[valid_mask]
        
        A_init = np.max(accuracies_valid)
        x0_init = np.median(layers_valid[accuracies_valid > 0.5 * A_init]) if any(accuracies_valid > 0.5 * A_init) else len(layers_valid) / 2
        k_init = 1.0
        
        try:
            popt, _ = curve_fit(
                self.logistic_function,
                layers_valid,
                accuracies_valid,
                p0=[A_init, k_init, x0_init],
                maxfev=5000,
                bounds=([0, 0, 0], [1.5, 10, len(layers)])
            )
            
            y_pred = self.logistic_function(layers_valid, *popt)
            ss_res = np.sum((accuracies_valid - y_pred) ** 2)
            ss_tot = np.sum((accuracies_valid - np.mean(accuracies_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return popt[1], r2
            
        except Exception as e:
            return np.nan, np.nan
    
    def find_crystallization_layer(self, layer_stats):
        """Find crystallization layer"""
        accuracy_threshold = 0.7
        stability_window = 3
        stability_epsilon = 0.05
        confidence_threshold = 0.01
        
        num_layers = len(layer_stats)
        skip_layers = max(3, num_layers // 10)
        
        for layer_idx in range(skip_layers, num_layers - stability_window + 1):
            if layer_stats[layer_idx]['accuracy'] < accuracy_threshold:
                continue
            
            window_accuracies = [layer_stats[i]['accuracy'] 
                               for i in range(layer_idx, min(layer_idx + stability_window, num_layers))]
            
            if all(acc >= accuracy_threshold - stability_epsilon for acc in window_accuracies):
                if layer_stats[layer_idx]['confidence'] >= confidence_threshold:
                    return layer_idx
        
        valid_layers = range(skip_layers, num_layers)
        if valid_layers:
            max_acc_idx = max(valid_layers, key=lambda i: layer_stats[i]['accuracy'])
            return max_acc_idx
        
        return num_layers - 1
    
    def calculate_binary_competence(self, layer_stats, cryst_layer):
        """Calculate pre-emergence and post-emergence accuracies"""
        if cryst_layer <= 0 or cryst_layer >= len(layer_stats):
            return np.nan, np.nan
        
        pre_accuracies = [layer_stats[i]['accuracy'] for i in range(cryst_layer)]
        pre_acc = np.mean(pre_accuracies) if pre_accuracies else np.nan
        
        post_accuracies = [layer_stats[i]['accuracy'] for i in range(cryst_layer, len(layer_stats))]
        post_acc = np.mean(post_accuracies) if post_accuracies else np.nan
        
        return pre_acc, post_acc
    
    def find_probing_file(self, model_dir):
        """Find the COMBINED probing CSV file"""
        probing_dir = model_dir / "probing"
        
        if not probing_dir.exists():
            return None
        
        # ✅ PRIORITIZE *_combined.csv files
        combined_files = list(probing_dir.glob("*_combined.csv"))
        if combined_files:
            return combined_files[0]
        
        # Fallback to regular probing files
        csv_files = list(probing_dir.glob("*_probing.csv"))
        if csv_files:
            return csv_files[0]
        
        return None
    
    def process_single_model_task(self, model_dir, dataset_name, model_display_name):
        """Extract all statistics for one model-task combination"""
        probing_file = self.find_probing_file(model_dir)
        
        if probing_file is None:
            return None
        
        df = pd.read_csv(probing_file)
        
        # Filter for this dataset
        df_task = df[df['dataset_name'] == dataset_name].copy()
        
        if len(df_task) == 0:
            return None
        
        num_layers = int(df_task['layer_idx'].max() + 1)
        
        # Calculate layer-wise statistics
        layer_stats = []
        for layer_idx in range(num_layers):
            layer_data = df_task[df_task['layer_idx'] == layer_idx]
            if len(layer_data) == 0:
                layer_stats.append({
                    'layer': layer_idx,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'n_samples': 0
                })
            else:
                layer_stats.append({
                    'layer': layer_idx,
                    'accuracy': float(layer_data['is_correct'].mean()),
                    'confidence': float(layer_data['layer_confidence'].mean()),
                    'n_samples': int(len(layer_data))
                })
        
        cryst_layer = self.find_crystallization_layer(layer_stats)
        norm_depth = (cryst_layer / num_layers) * 100 if num_layers > 0 else np.nan
        
        accuracies = [ls['accuracy'] for ls in layer_stats]
        k_value, r2_value = self.calculate_abruptness(accuracies)
        
        pre_acc, post_acc = self.calculate_binary_competence(layer_stats, cryst_layer)
        final_acc = layer_stats[-1]['accuracy'] if layer_stats else np.nan
        n_samples = int(df_task['sample_id'].nunique())
        
        result = {
            'model': model_display_name,
            'task': dataset_name,
            'num_layers': num_layers,
            'crystallization_layer': cryst_layer,
            'normalized_depth_pct': norm_depth,
            'abruptness_k': k_value,
            'fit_r2': r2_value,
            'pre_emergence_acc': pre_acc * 100 if not np.isnan(pre_acc) else np.nan,
            'post_emergence_acc': post_acc * 100 if not np.isnan(post_acc) else np.nan,
            'final_accuracy': final_acc * 100 if not np.isnan(final_acc) else np.nan,
            'accuracy_gap': (post_acc - pre_acc) * 100 if not np.isnan(pre_acc) and not np.isnan(post_acc) else np.nan,
            'n_samples': n_samples
        }
        
        return result
    
    def generate_full_table(self):
        """Generate complete statistics table"""
        print("\n" + "="*80)
        print("📊 GENERATING SUPPLEMENTARY TABLE S1 - ALL TASKS")
        print("="*80 + "\n")
        
        model_configs = [
            {'dir': 'llama_1b_models', 'name': 'Llama-1B', 'family': 'Llama'},
            {'dir': 'llama_3b_model', 'name': 'Llama-3B', 'family': 'Llama'},
            {'dir': 'llama_5b_model', 'name': 'Llama-5B', 'family': 'Llama'},
            {'dir': 'llama_13b_models', 'name': 'Llama-13B', 'family': 'Llama'},
            {'dir': 'phi_1_models', 'name': 'Phi-1.0', 'family': 'Phi'},
            {'dir': 'phi_1.5_models', 'name': 'Phi-1.5', 'family': 'Phi'},
            {'dir': 'deepsek_7b_models', 'name': 'DeepSeek-7B', 'family': 'DeepSeek'},
        ]
        
        tasks = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        
        all_results = []
        
        for model_config in model_configs:
            model_dir = self.base_dir / model_config['dir']
            
            print(f"\n🔍 {model_config['name']}")
            
            if not model_dir.exists():
                continue
            
            for task_name in tasks:
                result = self.process_single_model_task(model_dir, task_name, model_config['name'])
                
                if result is not None:
                    result['family'] = model_config['family']
                    all_results.append(result)
                    print(f"  ✅ {task_name:15s}: Layer {result['crystallization_layer']:2d}/{result['num_layers']:2d} ({result['normalized_depth_pct']:5.1f}%)")
        
        if not all_results:
            return None
        
        df = pd.DataFrame(all_results)
        
        task_order = {'gsm8k': 0, 'boolq': 1, 'commonsense_qa': 2, 'hellaswag': 3}
        df['task_order'] = df['task'].map(task_order)
        df = df.sort_values(['family', 'num_layers', 'task_order'])
        
        print("\n" + "="*80)
        print(f"✅ COMPLETE: {len(df)} model-task combinations")
        print("="*80)
        
        return df
    
    def format_latex_table(self, df):
        """Format LaTeX table"""
        if df is None or len(df) == 0:
            return "No data available"
        
        latex = []
        latex.append("\\begin{table*}[t]")
        latex.append("\\centering")
        latex.append("\\scriptsize")
        latex.append("\\caption{\\textbf{Comprehensive emergence statistics across all model-task combinations.}}")
        latex.append("\\label{tab:supp_stats}")
        latex.append("\\begin{tabular}{llccccccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Task} & \\textbf{Layers} & \\textbf{Cryst.} & \\textbf{Depth} & \\textbf{$k$} & \\textbf{$R^2$} & \\textbf{Pre} & \\textbf{Post} & \\textbf{Gap} & \\textbf{Final} \\\\")
        latex.append(" & & & \\textbf{Layer} & \\textbf{(\\%)} & & & \\textbf{(\\%)} & \\textbf{(\\%)} & \\textbf{(pp)} & \\textbf{(\\%)} \\\\")
        latex.append("\\midrule")
        
        current_family = None
        
        for idx, row in df.iterrows():
            if row['family'] != current_family:
                if current_family is not None:
                    latex.append("\\midrule")
                current_family = row['family']
            
            task_map = {'gsm8k': 'GSM8K', 'boolq': 'BoolQ', 'commonsense_qa': 'CommQA', 'hellaswag': 'HellaSwag'}
            task_short = task_map.get(row['task'], row['task'])
            
            vals = {
                'cryst': f"{row['crystallization_layer']:.0f}",
                'norm': f"{row['normalized_depth_pct']:.0f}",
                'k': f"{row['abruptness_k']:.2f}" if not np.isnan(row['abruptness_k']) else "---",
                'r2': f"{row['fit_r2']:.2f}" if not np.isnan(row['fit_r2']) else "---",
                'pre': f"{row['pre_emergence_acc']:.0f}" if not np.isnan(row['pre_emergence_acc']) else "---",
                'post': f"{row['post_emergence_acc']:.0f}" if not np.isnan(row['post_emergence_acc']) else "---",
                'gap': f"{row['accuracy_gap']:.0f}" if not np.isnan(row['accuracy_gap']) else "---",
                'final': f"{row['final_accuracy']:.0f}" if not np.isnan(row['final_accuracy']) else "---"
            }
            
            latex.append(f"{row['model']} & {task_short} & {row['num_layers']} & {vals['cryst']} & {vals['norm']} & {vals['k']} & {vals['r2']} & {vals['pre']} & {vals['post']} & {vals['gap']} & {vals['final']} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table*}")
        
        return "\n".join(latex)
    
    def save_results(self, df):
        """Save results"""
        if df is None or len(df) == 0:
            return
        
        output_dir = self.base_dir.parent
        
        output_csv = output_dir / "supplementary_table_s1.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n💾 CSV: {output_csv}")
        
        latex_content = self.format_latex_table(df)
        output_tex = output_dir / "supplementary_table_s1.tex"
        with open(output_tex, 'w') as f:
            f.write(latex_content)
        print(f"💾 LaTeX: {output_tex}")
        
        print("\n" + "="*80)
        print("📊 SUMMARY STATISTICS")
        print("="*80)
        
        print("\n🔹 By Family:")
        print(df.groupby('family')[['normalized_depth_pct', 'abruptness_k', 'accuracy_gap', 'final_accuracy']].agg(['mean', 'std']).round(1))
        
        print("\n🔹 By Task:")
        print(df.groupby('task')[['normalized_depth_pct', 'abruptness_k', 'accuracy_gap', 'final_accuracy']].agg(['mean', 'std']).round(1))
        
        print("\n🎯 KEY FINDINGS:")
        print(f"📍 Earliest: {df.loc[df['normalized_depth_pct'].idxmin()]['model']} on {df.loc[df['normalized_depth_pct'].idxmin()]['task']} ({df['normalized_depth_pct'].min():.0f}%)")
        print(f"📍 Latest: {df.loc[df['normalized_depth_pct'].idxmax()]['model']} on {df.loc[df['normalized_depth_pct'].idxmax()]['task']} ({df['normalized_depth_pct'].max():.0f}%)")
        print(f"⚡ Most abrupt: k={df['abruptness_k'].max():.2f}")
        print(f"📊 Largest gap: {df['accuracy_gap'].max():.0f} pp")


if __name__ == "__main__":
    generator = SupplementaryTableGenerator(base_dir="E:/publication/ed/viz")
    df = generator.generate_full_table()
    
    if df is not None:
        generator.save_results(df)
        print("\n✅ DONE! Now paste the results to me!")
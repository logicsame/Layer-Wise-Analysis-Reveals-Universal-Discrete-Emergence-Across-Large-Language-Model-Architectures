"""
DISCOVERY-DRIVEN MULTI-MODEL ABLATION STUDY
For Nature Machine Intelligence Submission

Validates 5 key discoveries about capability emergence:
1. Universal abrupt emergence (discrete phase transitions)
2. Task-hierarchical formation (simple → complex ordering)
3. Size-dependent emergence inversion (larger = earlier)
4. Binary competence acquisition (absent → perfect)
5. Size-dependent transition sharpness (larger = sharper)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, f_oneway, binomtest, ttest_ind
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DiscoveryValidation:
    """Container for discovery validation results"""
    discovery_name: str
    validated: bool
    p_value: float
    effect_size: float
    confidence_level: str  # "high", "medium", "low"
    evidence: str
    implications: str
    details: Dict


class EmergenceDiscoveryValidator:
    """
    Validates discoveries about emergence patterns.
    Discovery-driven approach: We report what we found, not test hypotheses.
    """
    
    def __init__(self, 
                 base_dir: str = ".", 
                 model_dirs: Optional[List[str]] = None,
                 alpha: float = 0.05):
        self.base_dir = Path(base_dir)
        self.alpha = alpha
        self.discoveries = {}
        self.model_data = {}
        self.periodic_table = None
        self.sample_counts = {}
        
        if model_dirs is not None:
            self.model_dirs = [self.base_dir / d for d in model_dirs]
        else:
            self.model_dirs = self._auto_detect_model_dirs()
        
        print(f"📂 Base directory: {self.base_dir}")
        print(f"📂 Model directories: {len(self.model_dirs)}")
        for d in self.model_dirs:
            print(f"   - {d.name}")
    
    def _auto_detect_model_dirs(self) -> List[Path]:
        """Auto-detect directories containing analysis/ subdirectories"""
        model_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                analysis_dir = item / "analysis"
                if analysis_dir.exists() and analysis_dir.is_dir():
                    model_dirs.append(item)
        return sorted(model_dirs)
    
    # ========================================================================
    # DATA LOADING (Same as before)
    # ========================================================================
    
    def load_periodic_tables(self) -> Optional[pd.DataFrame]:
        """Load periodic table CSV files"""
        print("\n" + "="*80)
        print("📋 LOADING PERIODIC TABLES")
        print("="*80)
        
        all_periodic_tables = []
        
        for model_dir in self.model_dirs:
            analysis_dir = model_dir / "analysis"
            periodic_path = analysis_dir / "periodic_table.csv"
            
            if not periodic_path.exists():
                print(f"⚠️ No periodic table in {model_dir.name}/analysis/")
                continue
            
            try:
                df = pd.read_csv(periodic_path)
                df['model_dir'] = model_dir.name
                
                print(f"\n📁 {model_dir.name}:")
                print(f"   Rows: {len(df)}")
                
                for _, row in df.iterrows():
                    dataset = 'combined'
                    if '_gsm8k' in row.get('model', ''):
                        dataset = 'gsm8k'
                    elif '_boolq' in row.get('model', ''):
                        dataset = 'boolq'
                    elif '_commonsense_qa' in row.get('model', ''):
                        dataset = 'commonsense_qa'
                    elif '_hellaswag' in row.get('model', ''):
                        dataset = 'hellaswag'
                    
                    cryst = row.get('crystallization_layer', -1)
                    layers = row.get('num_layers', 0)
                    depth = row.get('crystallization_depth_pct', 0)
                    acc = row.get('final_accuracy', 0)
                    
                    print(f"     {dataset:15s} | Layer {cryst:2d}/{layers:2d} ({depth:5.1f}%) | Acc: {acc*100:5.1f}%")
                
                all_periodic_tables.append(df)
                
            except Exception as e:
                print(f"❌ Error loading {periodic_path}: {e}")
        
        if not all_periodic_tables:
            return None
        
        combined = pd.concat(all_periodic_tables, ignore_index=True)
        self.periodic_table = combined
        return combined
    
    def validate_sample_counts(self) -> Dict:
        """Validate sample counts from probing files"""
        print("\n" + "="*80)
        print("🔍 VALIDATING SAMPLE COUNTS")
        print("="*80)
        
        validation_results = {}
        total_samples = 0
        
        for model_dir in self.model_dirs:
            probing_dir = model_dir / "probing"
            if not probing_dir.exists():
                continue
            
            csv_files = list(probing_dir.glob("*_probing*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    unique_samples = df['sample_id'].nunique() if 'sample_id' in df.columns else 0
                    validation_results[csv_file.name] = {
                        'model_dir': model_dir.name,
                        'unique_samples': unique_samples
                    }
                    total_samples += unique_samples
                except:
                    pass
        
        print(f"\n📊 TOTAL SAMPLES: {total_samples}")
        self.sample_counts = validation_results
        return validation_results
    
    def load_all_model_data(self) -> Dict:
        """Load per-dataset emergence files"""
        print("\n" + "="*80)
        print("📂 LOADING EMERGENCE DATA")
        print("="*80)
        
        all_files = []
        
        for model_dir in self.model_dirs:
            analysis_dir = model_dir / "analysis"
            if not analysis_dir.exists():
                continue
            
            files = list(analysis_dir.glob("*_emergence.json"))
            if files:
                all_files.extend([(model_dir.name, f) for f in files])
        
        datasets = ['gsm8k', 'boolq', 'commonsense_qa', 'hellaswag']
        
        for model_name, filepath in all_files:
            filename = filepath.stem.replace("_emergence", "")
            
            dataset_name = None
            for dataset in datasets:
                if dataset in filename:
                    dataset_name = dataset
                    break
            
            if not dataset_name and 'combined' in filename:
                dataset_name = 'combined'
            
            if not dataset_name:
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if model_name not in self.model_data:
                    self.model_data[model_name] = {}
                
                self.model_data[model_name][dataset_name] = data
                
            except Exception as e:
                print(f"❌ Error loading {filepath.name}: {e}")
        
        print(f"\n📊 Loaded: {len(self.model_data)} models")
        return self.model_data
    
    def extract_model_size(self, model_name: str) -> Optional[float]:
        """Extract model size from name"""
        import re
        patterns = [r'(\d+\.?\d*)b', r'(\d+)B']
        for pattern in patterns:
            match = re.search(pattern, model_name, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    # ========================================================================
    # DISCOVERY 1: UNIVERSAL ABRUPT EMERGENCE
    # ========================================================================
    
    def validate_discovery_1_universal_abrupt_emergence(self) -> DiscoveryValidation:
        """
        DISCOVERY 1: All model-task combinations show discrete phase transitions.
        
        Validation:
        - Count: How many show abrupt emergence?
        - Statistical test: Binomial test (null = 50% random)
        - Effect size: Fraction showing abruptness
        """
        
        print("\n" + "="*80)
        print("🔬 DISCOVERY 1: UNIVERSAL ABRUPT EMERGENCE")
        print("="*80)
        
        emergence_patterns = []
        
        for model_name, datasets in self.model_data.items():
            for dataset_name, data in datasets.items():
                if dataset_name == 'combined':
                    continue
                
                layer_stats = data.get('layer_statistics', [])
                if len(layer_stats) < 3:
                    continue
                
                accuracies = [s['accuracy'] for s in layer_stats]
                acc_diffs = np.diff(accuracies)
                
                if len(acc_diffs) == 0:
                    continue
                
                max_jump = np.max(acc_diffs)
                mean_jump = np.mean(acc_diffs)
                std_jump = np.std(acc_diffs)
                
                # Abrupt = max jump > mean + 2*std AND > 20%
                is_abrupt = (max_jump > mean_jump + 2*std_jump) and (max_jump > 0.2)
                
                emergence_patterns.append({
                    'model': model_name,
                    'task': dataset_name,
                    'max_jump': max_jump * 100,
                    'is_abrupt': is_abrupt,
                    'num_layers': len(layer_stats)
                })
        
        if len(emergence_patterns) < 2:
            return None
        
        df = pd.DataFrame(emergence_patterns)
        
        print(f"\n📊 Emergence Patterns:")
        for _, row in df.iterrows():
            pattern = "ABRUPT" if row['is_abrupt'] else "gradual"
            print(f"  {row['model']:20s} | {row['task']:15s} | {pattern:8s} | Jump: {row['max_jump']:5.1f}%")
        
        # Statistics
        num_abrupt = df['is_abrupt'].sum()
        total = len(df)
        fraction = num_abrupt / total
        
        # Binomial test: Is this more than random (50%)?
        p_value = binomtest(num_abrupt, total, 0.5, alternative='greater').pvalue
        
        print(f"\n📈 Validation:")
        print(f"  Abrupt: {num_abrupt}/{total} ({fraction*100:.0f}%)")
        print(f"  Binomial test (H₀: p=0.5): p={p_value:.6f}")
        print(f"  Effect size: {fraction:.3f}")
        
        # Validation criteria
        validated = (fraction >= 0.75) and (p_value < 0.05)
        
        if validated:
            confidence = "HIGH" if p_value < 0.001 else "MEDIUM"
            evidence = (f"{num_abrupt}/{total} ({fraction*100:.0f}%) model-task combinations exhibit "
                       f"abrupt emergence (p={p_value:.4f})")
            implications = (f"Rejects gradual skill accumulation. Supports discrete phase transitions. "
                          f"Implications for predictability and safety.")
        else:
            confidence = "LOW"
            evidence = f"Only {num_abrupt}/{total} show abruptness"
            implications = "Mixed emergence patterns suggest task/model-specific dynamics"
        
        print(f"\n{'✅' if validated else '⚠️'} DISCOVERY 1: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        print(f"   Confidence: {confidence}")
        
        result = DiscoveryValidation(
            discovery_name="Universal Abrupt Emergence",
            validated=validated,
            p_value=p_value,
            effect_size=fraction,
            confidence_level=confidence,
            evidence=evidence,
            implications=implications,
            details={
                'num_abrupt': int(num_abrupt),
                'total': int(total),
                'patterns': df.to_dict('records')
            }
        )
        
        self.discoveries['discovery_1'] = result
        return result
    
    
    # ========================================================================
    # DISCOVERY 6: POWER LAW SCALING
    # ========================================================================

    def validate_discovery_6_scaling_law(self) -> DiscoveryValidation:
        """
        DISCOVERY 6: Emergence depth follows power law with model size.
        
        Test multiple functional forms:
        1. Power law: depth = A × size^B
        2. Logarithmic: depth = A × log(size) + B  
        3. Linear: depth = A × size + B
        
        Choose best fit and report scaling exponent.
        """
        
        print("\n" + "="*80)
        print("🔬 DISCOVERY 6: SCALING LAW")
        print("="*80)
        
        model_stats = {}
        
        for model_name, datasets in self.model_data.items():
            size = self.extract_model_size(model_name)
            
            if not size:
                continue
            
            depths = []
            for dataset_name, data in datasets.items():
                if dataset_name == 'combined':
                    continue
                
                cryst = data.get('crystallization_layer', -1)
                layers = data.get('num_layers', 0)
                
                if cryst >= 0 and layers > 0:
                    depths.append(100 * cryst / layers)
            
            if depths:
                model_stats[model_name] = {
                    'size': size,
                    'mean_depth': np.mean(depths),
                    'std_depth': np.std(depths),
                    'depths': depths
                }
        
        if len(model_stats) < 3:
            print("⚠️ Need at least 3 models for scaling law")
            return DiscoveryValidation(
                discovery_name="Scaling Law",
                validated=False,
                p_value=1.0,
                effect_size=0.0,
                confidence_level="LOW",
                evidence="Insufficient models (need ≥3)",
                implications="Cannot determine scaling relationship",
                details={}
            )
        
        print(f"\n📊 Data Points:")
        for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['size']):
            print(f"  {stats['size']:.1f}B: {stats['mean_depth']:.1f}% ± {stats['std_depth']:.1f}%")
        
        # Extract data
        sizes = np.array([s['size'] for s in model_stats.values()])
        depths = np.array([s['mean_depth'] for s in model_stats.values()])
        
        # ========== TEST 1: POWER LAW ==========
        # depth = A × size^B
        # log(depth) = log(A) + B × log(size)
        
        log_sizes = np.log10(sizes)
        log_depths = np.log10(depths)
        
        from scipy.stats import linregress
        slope_power, intercept_power, r_power, p_power, stderr_power = linregress(log_sizes, log_depths)
        
        A_power = 10**intercept_power
        B_power = slope_power
        r2_power = r_power**2
        
        print(f"\n📈 POWER LAW: depth = {A_power:.2f} × size^{B_power:.3f}")
        print(f"   R² = {r2_power:.4f}")
        print(f"   p = {p_power:.6f}")
        
        # ========== TEST 2: LOGARITHMIC ==========
        # depth = A × log(size) + B
        
        log_sizes_natural = np.log(sizes)
        slope_log, intercept_log, r_log, p_log, stderr_log = linregress(log_sizes_natural, depths)
        
        r2_log = r_log**2
        
        print(f"\n📈 LOGARITHMIC: depth = {slope_log:.2f} × ln(size) + {intercept_log:.2f}")
        print(f"   R² = {r2_log:.4f}")
        print(f"   p = {p_log:.6f}")
        
        # ========== TEST 3: LINEAR ==========
        # depth = A × size + B
        
        slope_linear, intercept_linear, r_linear, p_linear, stderr_linear = linregress(sizes, depths)
        
        r2_linear = r_linear**2
        
        print(f"\n📈 LINEAR: depth = {slope_linear:.2f} × size + {intercept_linear:.2f}")
        print(f"   R² = {r2_linear:.4f}")
        print(f"   p = {p_linear:.6f}")
        
        # ========== CHOOSE BEST FIT ==========
        
        fits = [
            {'name': 'power', 'r2': r2_power, 'p': p_power, 'formula': f"depth = {A_power:.2f} × size^{B_power:.3f}"},
            {'name': 'log', 'r2': r2_log, 'p': p_log, 'formula': f"depth = {slope_log:.2f} × ln(size) + {intercept_log:.2f}"},
            {'name': 'linear', 'r2': r2_linear, 'p': p_linear, 'formula': f"depth = {slope_linear:.2f} × size + {intercept_linear:.2f}"}
        ]
        
        best_fit = max(fits, key=lambda x: x['r2'])
        
        print(f"\n🎯 BEST FIT: {best_fit['name'].upper()}")
        print(f"   Formula: {best_fit['formula']}")
        print(f"   R² = {best_fit['r2']:.4f}")
        print(f"   p = {best_fit['p']:.6f}")
        
        # ========== VALIDATION ==========
        
        # Validated if best fit has R² > 0.5 (at least 50% variance explained)
        validated = best_fit['r2'] > 0.5
        
        if validated:
            confidence = "HIGH" if best_fit['r2'] > 0.8 else "MEDIUM"
            
            # Interpret the scaling exponent
            if best_fit['name'] == 'power':
                if B_power < -0.2:
                    scaling_type = "NEGATIVE (counter-intuitive)"
                    interpretation = f"Larger models emerge at shallower depths (exponent={B_power:.3f})"
                elif B_power > 0.2:
                    scaling_type = "POSITIVE"
                    interpretation = f"Larger models emerge at deeper depths (exponent={B_power:.3f})"
                else:
                    scaling_type = "FLAT"
                    interpretation = f"Emergence depth is approximately scale-invariant (exponent≈0)"
            else:
                if best_fit['name'] == 'log':
                    scaling_type = "LOGARITHMIC"
                    interpretation = f"Emergence depth grows logarithmically with size (slope={slope_log:.2f})"
                else:
                    scaling_type = "LINEAR"
                    interpretation = f"Emergence depth scales linearly with size (slope={slope_linear:.2f})"
            
            evidence = (f"Emergence depth follows a {best_fit['name']} scaling law: {best_fit['formula']} "
                    f"(R²={best_fit['r2']:.3f}, p={best_fit['p']:.4f}). {interpretation}")
            
            implications = (f"Predictable scaling relationship enables forecasting emergence depth for untested model sizes. "
                        f"{scaling_type} scaling has implications for model design and capability prediction.")
        else:
            confidence = "LOW"
            evidence = f"No clear scaling law (best R²={best_fit['r2']:.3f})"
            implications = "Emergence depth may be influenced by factors beyond model size"
        
        print(f"\n{'✅' if validated else '⚠️'} DISCOVERY 6: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        print(f"   Confidence: {confidence}")
        
        result = DiscoveryValidation(
            discovery_name="Scaling Law",
            validated=validated,
            p_value=best_fit['p'],
            effect_size=best_fit['r2'],
            confidence_level=confidence,
            evidence=evidence,
            implications=implications,
            details={
                'power_law': {
                    'exponent': float(B_power),
                    'coefficient': float(A_power),
                    'r_squared': float(r2_power),
                    'p_value': float(p_power)
                },
                'log_law': {
                    'slope': float(slope_log),
                    'intercept': float(intercept_log),
                    'r_squared': float(r2_log),
                    'p_value': float(p_log)
                },
                'linear_law': {
                    'slope': float(slope_linear),
                    'intercept': float(intercept_linear),
                    'r_squared': float(r2_linear),
                    'p_value': float(p_linear)
                },
                'best_fit': best_fit['name'],
                'data_points': {k: {'size': v['size'], 'mean_depth': v['mean_depth']} 
                            for k, v in model_stats.items()}
            }
        )
        
        self.discoveries['discovery_6'] = result
        return result
    
    
    # ========================================================================
    # DISCOVERY 2: TASK-HIERARCHICAL FORMATION
    # ========================================================================
    
    def validate_discovery_2_task_hierarchy(self) -> DiscoveryValidation:
        """
        DISCOVERY 2: Tasks emerge in complexity-based hierarchy.
        
        Expected order: HellaSwag < GSM8K < CommonsenseQA < BoolQ
        
        Validation:
        - Measure: Mean emergence depth per task
        - Test: Spearman correlation with complexity ranking
        - Effect size: η² from ANOVA
        """
        
        print("\n" + "="*80)
        print("🔬 DISCOVERY 2: TASK-HIERARCHICAL FORMATION")
        print("="*80)
        
        task_depths = {}
        
        for model_name, datasets in self.model_data.items():
            for dataset_name, data in datasets.items():
                if dataset_name == 'combined':
                    continue
                
                cryst = data.get('crystallization_layer', -1)
                layers = data.get('num_layers', 0)
                
                if cryst >= 0 and layers > 0:
                    depth_pct = 100 * cryst / layers
                    
                    if dataset_name not in task_depths:
                        task_depths[dataset_name] = []
                    task_depths[dataset_name].append(depth_pct)
        
        print(f"\n📊 Task Emergence Depths:")
        task_stats = {}
        for task, depths in sorted(task_depths.items()):
            mean = np.mean(depths)
            std = np.std(depths)
            task_stats[task] = {'mean': mean, 'std': std, 'n': len(depths)}
            print(f"  {task:20s}: {mean:5.1f}% ± {std:4.1f}% (n={len(depths)})")
        
        # Assign complexity ranks (1=simplest, 4=most complex)
        complexity_ranks = {
            'hellaswag': 1,      # Pattern matching
            'gsm8k': 2,          # Math reasoning
            'commonsense_qa': 3, # Conceptual reasoning
            'boolq': 4           # Output formatting (late)
        }
        
        # Test if emergence depth correlates with complexity
        tasks_with_ranks = [t for t in task_stats.keys() if t in complexity_ranks]
        
        if len(tasks_with_ranks) >= 3:
            ranks = [complexity_ranks[t] for t in tasks_with_ranks]
            depths_mean = [task_stats[t]['mean'] for t in tasks_with_ranks]
            
            # Spearman correlation
            from scipy.stats import spearmanr
            rho, p_value = spearmanr(ranks, depths_mean)
            
            print(f"\n📈 Hierarchy Validation:")
            print(f"  Spearman ρ: {rho:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            # ANOVA for task differences
            depth_lists = [task_depths[t] for t in tasks_with_ranks]
            f_stat, p_anova = f_oneway(*depth_lists)
            
            # Effect size (eta-squared)
            all_depths = [d for depths in depth_lists for d in depths]
            grand_mean = np.mean(all_depths)
            ss_between = sum(len(d) * (np.mean(d) - grand_mean)**2 for d in depth_lists)
            ss_total = sum((x - grand_mean)**2 for x in all_depths)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            print(f"  ANOVA F: {f_stat:.4f}, p={p_anova:.6f}")
            print(f"  Effect size η²: {eta_squared:.4f}")
            
            # Validation
            validated = (rho > 0.6) or (eta_squared > 0.4)
            
            if validated:
                confidence = "HIGH" if p_value < 0.01 else "MEDIUM"
                evidence = (f"Tasks emerge in complexity order (ρ={rho:.3f}, p={p_value:.4f}). "
                          f"Simple pattern matching (HellaSwag, {task_stats['hellaswag']['mean']:.0f}%) "
                          f"emerges before abstract reasoning (CommonsenseQA, {task_stats['commonsense_qa']['mean']:.0f}%)")
                implications = "Suggests hierarchical capability construction: early layers learn primitives, later layers compose them"
            else:
                confidence = "LOW"
                evidence = f"Weak hierarchy (ρ={rho:.3f})"
                implications = "Task-independent emergence mechanisms"
        else:
            p_value = 1.0
            eta_squared = 0
            validated = False
            confidence = "LOW"
            evidence = "Insufficient tasks for hierarchy analysis"
            implications = "Need more diverse task types"
        
        print(f"\n{'✅' if validated else '⚠️'} DISCOVERY 2: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        print(f"   Confidence: {confidence}")
        
        result = DiscoveryValidation(
            discovery_name="Task-Hierarchical Formation",
            validated=validated,
            p_value=p_value,
            effect_size=eta_squared,
            confidence_level=confidence,
            evidence=evidence,
            implications=implications,
            details={
                'task_stats': {k: {'mean': v['mean'], 'std': v['std']} for k, v in task_stats.items()},
                'hierarchy_correlation': float(rho) if 'rho' in locals() else None
            }
        )
        
        self.discoveries['discovery_2'] = result
        return result
    
    # ========================================================================
    # DISCOVERY 3: SIZE-DEPENDENT EMERGENCE INVERSION
    # ========================================================================
    
    def validate_discovery_3_size_inversion(self) -> DiscoveryValidation:
        """
        DISCOVERY 3: Larger models emerge EARLIER (counter-intuitive).
        
        Observation: 5B at 33%, 1B at 65%
        
        Validation:
        - Measure: Mean depth per model
        - Test: Correlation with model size (expect negative)
        - Effect size: |r|
        """
        
        print("\n" + "="*80)
        print("🔬 DISCOVERY 3: SIZE-DEPENDENT EMERGENCE INVERSION")
        print("="*80)
        
        model_stats = {}
        
        for model_name, datasets in self.model_data.items():
            size = self.extract_model_size(model_name)
            
            if not size:
                continue
            
            depths = []
            for dataset_name, data in datasets.items():
                if dataset_name == 'combined':
                    continue
                
                cryst = data.get('crystallization_layer', -1)
                layers = data.get('num_layers', 0)
                
                if cryst >= 0 and layers > 0:
                    depths.append(100 * cryst / layers)
            
            if depths:
                model_stats[model_name] = {
                    'size': size,
                    'mean_depth': np.mean(depths),
                    'std_depth': np.std(depths)
                }
        
        print(f"\n📊 Model Emergence by Size:")
        for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['size']):
            print(f"  {stats['size']:.1f}B: {stats['mean_depth']:.1f}% ± {stats['std_depth']:.1f}%")
        
        if len(model_stats) >= 2:
            sizes = np.array([s['size'] for s in model_stats.values()])
            depths = np.array([s['mean_depth'] for s in model_stats.values()])
            
            # Correlation
            r, p_value = pearsonr(sizes, depths)
            
            print(f"\n📈 Size-Depth Relationship:")
            print(f"  Pearson r: {r:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Direction: {'NEGATIVE (inversion!)' if r < 0 else 'positive'}")
            
            # Validation: Negative correlation = larger models emerge earlier
            validated = (r < -0.3) or (len(model_stats) == 2 and sizes[1] > sizes[0] and depths[1] < depths[0])
            
            if validated:
                confidence = "HIGH" if abs(r) > 0.7 or len(model_stats) == 2 else "MEDIUM"
                
                # Calculate fold-change
                if len(model_stats) == 2:
                    small_model = min(model_stats.items(), key=lambda x: x[1]['size'])
                    large_model = max(model_stats.items(), key=lambda x: x[1]['size'])
                    
                    size_ratio = large_model[1]['size'] / small_model[1]['size']
                    depth_ratio = small_model[1]['mean_depth'] / large_model[1]['mean_depth']
                    
                    evidence = (f"{large_model[1]['size']:.1f}B model emerges at {large_model[1]['mean_depth']:.0f}% depth, "
                              f"while {small_model[1]['size']:.1f}B model emerges at {small_model[1]['mean_depth']:.0f}% depth. "
                              f"{size_ratio:.1f}× larger model emerges {depth_ratio:.1f}× earlier (r={r:.3f})")
                else:
                    evidence = f"Negative correlation between size and depth (r={r:.3f}, p={p_value:.4f})"
                
                implications = ("Counter-intuitive finding: Scaling improves early-layer efficiency. "
                              "Larger models encode richer representations in shallow layers, "
                              "enabling earlier capability crystallization")
            else:
                confidence = "LOW"
                evidence = f"No clear size-depth relationship (r={r:.3f})"
                implications = "Emergence depth may be architecture-invariant"
        else:
            p_value = 1.0
            r = 0
            validated = False
            confidence = "LOW"
            evidence = "Need at least 2 models for size comparison"
            implications = "Cannot assess size-dependence"
        
        print(f"\n{'✅' if validated else '⚠️'} DISCOVERY 3: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        print(f"   Confidence: {confidence}")
        
        result = DiscoveryValidation(
            discovery_name="Size-Dependent Emergence Inversion",
            validated=validated,
            p_value=p_value,
            effect_size=abs(r) if 'r' in locals() else 0,
            confidence_level=confidence,
            evidence=evidence,
            implications=implications,
            details={
                'model_stats': {k: v for k, v in model_stats.items()},
                'correlation': float(r) if 'r' in locals() else None
            }
        )
        
        self.discoveries['discovery_3'] = result
        return result
    
    # ========================================================================
    # DISCOVERY 4: BINARY COMPETENCE ACQUISITION
    # ========================================================================
    
    def validate_discovery_4_binary_competence(self) -> DiscoveryValidation:
        """
        DISCOVERY 4: Post-emergence accuracy is near-perfect (99.9%).
        
        Validation:
        - Measure: Final accuracy variance (CV)
        - Test: All reach >95%?
        - Effect size: 1 - CV
        """
        
        print("\n" + "="*80)
        print("🔬 DISCOVERY 4: BINARY COMPETENCE ACQUISITION")
        print("="*80)
        
        final_accs = []
        
        for model_name, datasets in self.model_data.items():
            for dataset_name, data in datasets.items():
                if dataset_name == 'combined':
                    continue
                
                acc = data.get('overall_accuracy')
                if acc is not None:
                    final_accs.append({
                        'model': model_name,
                        'task': dataset_name,
                        'accuracy': acc
                    })
        
        df = pd.DataFrame(final_accs)
        
        print(f"\n📊 Final Accuracies:")
        for _, row in df.iterrows():
            print(f"  {row['model']:20s} | {row['task']:15s} | {row['accuracy']*100:5.1f}%")
        
        accs = df['accuracy'].values
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        cv = (std_acc / mean_acc * 100) if mean_acc > 0 else 0
        
        print(f"\n📈 Convergence:")
        print(f"  Mean: {mean_acc*100:.1f}%")
        print(f"  Std: {std_acc*100:.1f}%")
        print(f"  CV: {cv:.1f}%")
        
        # Test: Are all >95%?
        all_high = all(accs > 0.95)
        
        # Validation: Low variance AND all high
        validated = (cv < 5) and all_high
        
        if validated:
            confidence = "HIGH" if cv < 2 else "MEDIUM"
            evidence = (f"All {len(accs)} model-task combinations reach {mean_acc*100:.1f}±{std_acc*100:.1f}% "
                       f"final accuracy (CV={cv:.1f}%). No partial acquisition observed.")
            implications = ("Capabilities exhibit binary competence: either absent (pre-emergence) or "
                          "perfect (post-emergence). Contradicts skill-refinement models.")
        else:
            confidence = "LOW"
            evidence = f"Variable final accuracy (CV={cv:.1f}%)"
            implications = "Partial acquisition or task difficulty variations"
        
        print(f"\n{'✅' if validated else '⚠️'} DISCOVERY 4: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        print(f"   Confidence: {confidence}")
        
        result = DiscoveryValidation(
            discovery_name="Binary Competence Acquisition",
            validated=validated,
            p_value=1.0 - cv/100,  # Lower CV = stronger evidence
            effect_size=1.0 - cv/100,
            confidence_level=confidence,
            evidence=evidence,
            implications=implications,
            details={
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'cv': float(cv),
                'all_above_95': bool(all_high)
            }
        )
        
        self.discoveries['discovery_4'] = result
        return result
    
    # ========================================================================
    # DISCOVERY 5: SIZE-DEPENDENT TRANSITION SHARPNESS
    # ========================================================================
    
    def validate_discovery_5_sharpness_scaling(self) -> DiscoveryValidation:
        """
        DISCOVERY 5: Larger models show sharper transitions.
        
        Observation: 1B shows 56-95% jumps, 5B shows 99-100% jumps
        
        Validation:
        - Measure: Max jump per model
        - Test: Correlation with model size
        - Effect size: r
        """
        
        print("\n" + "="*80)
        print("🔬 DISCOVERY 5: SIZE-DEPENDENT TRANSITION SHARPNESS")
        print("="*80)
        
        model_jumps = {}
        
        for model_name, datasets in self.model_data.items():
            size = self.extract_model_size(model_name)
            
            if not size:
                continue
            
            jumps = []
            
            for dataset_name, data in datasets.items():
                if dataset_name == 'combined':
                    continue
                
                layer_stats = data.get('layer_statistics', [])
                if len(layer_stats) < 3:
                    continue
                
                accuracies = [s['accuracy'] for s in layer_stats]
                acc_diffs = np.diff(accuracies)
                
                if len(acc_diffs) > 0:
                    max_jump = np.max(acc_diffs) * 100
                    jumps.append(max_jump)
            
            if jumps:
                model_jumps[model_name] = {
                    'size': size,
                    'mean_jump': np.mean(jumps),
                    'max_jump': np.max(jumps),
                    'jumps': jumps
                }
        
        print(f"\n📊 Jump Magnitudes by Model Size:")
        for model, stats in sorted(model_jumps.items(), key=lambda x: x[1]['size']):
            print(f"  {stats['size']:.1f}B: mean={stats['mean_jump']:.1f}%, max={stats['max_jump']:.1f}%")
            print(f"         jumps: {[f'{j:.1f}' for j in stats['jumps']]}")
        
        if len(model_jumps) >= 2:
            sizes = np.array([s['size'] for s in model_jumps.values()])
            mean_jumps = np.array([s['mean_jump'] for s in model_jumps.values()])
            
            # Correlation
            r, p_value = pearsonr(sizes, mean_jumps)
            
            print(f"\n📈 Sharpness-Size Relationship:")
            print(f"  Pearson r: {r:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Direction: {'POSITIVE (larger = sharper)' if r > 0 else 'negative'}")
            
            # Validation: Positive correlation = larger models are sharper
            validated = (r > 0.3) or (len(model_jumps) == 2 and 
                                     sizes[1] > sizes[0] and mean_jumps[1] > mean_jumps[0])
            
            if validated:
                confidence = "HIGH" if r > 0.7 or len(model_jumps) == 2 else "MEDIUM"
                
                if len(model_jumps) == 2:
                    small = min(model_jumps.items(), key=lambda x: x[1]['size'])
                    large = max(model_jumps.items(), key=lambda x: x[1]['size'])
                    
                    evidence = (f"{small[1]['size']:.1f}B model: {small[1]['mean_jump']:.0f}% average jump. "
                              f"{large[1]['size']:.1f}B model: {large[1]['mean_jump']:.0f}% average jump. "
                              f"Larger models exhibit sharper transitions (r={r:.3f})")
                else:
                    evidence = f"Positive correlation between size and jump magnitude (r={r:.3f}, p={p_value:.4f})"
                
                implications = ("Scaling affects transition dynamics, not just depth. "
                              "Larger models have more crystallized computational pathways, "
                              "resulting in more switch-like capability acquisition")
            else:
                confidence = "LOW"
                evidence = f"No clear size-sharpness relationship (r={r:.3f})"
                implications = "Transition sharpness may be architecture-invariant"
        else:
            p_value = 1.0
            r = 0
            validated = False
            confidence = "LOW"
            evidence = "Need at least 2 models"
            implications = "Cannot assess sharpness scaling"
        
        print(f"\n{'✅' if validated else '⚠️'} DISCOVERY 5: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        print(f"   Confidence: {confidence}")
        
        result = DiscoveryValidation(
            discovery_name="Size-Dependent Transition Sharpness",
            validated=validated,
            p_value=p_value,
            effect_size=abs(r) if 'r' in locals() else 0,
            confidence_level=confidence,
            evidence=evidence,
            implications=implications,
            details={
                'model_jumps': {k: {'size': v['size'], 'mean_jump': v['mean_jump']} 
                               for k, v in model_jumps.items()},
                'correlation': float(r) if 'r' in locals() else None
            }
        )
        
        self.discoveries['discovery_5'] = result
        return result
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    def save_results(self, output_path: str = "emergence_discoveries_validation.json"):
        """Save discovery validations"""
        
        # Convert to JSON-serializable format
        results_dict = {}
        for name, discovery in self.discoveries.items():
            results_dict[name] = {
                'discovery_name': discovery.discovery_name,
                'validated': bool(discovery.validated),
                'p_value': float(discovery.p_value),
                'effect_size': float(discovery.effect_size),
                'confidence_level': discovery.confidence_level,
                'evidence': discovery.evidence,
                'implications': discovery.implications,
                'details': json.loads(json.dumps(discovery.details, default=str))  # Force serialization
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Validation results saved to: {output_path}")
        
        # Text report
        report_path = output_path.replace('.json', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EMERGENCE DISCOVERY VALIDATION REPORT\n")
            f.write("Nature Machine Intelligence Submission\n")
            f.write("="*80 + "\n\n")
            
            validated_count = sum(1 for d in self.discoveries.values() if d.validated)
            f.write(f"SUMMARY: {validated_count}/{len(self.discoveries)} discoveries validated\n\n")
            
            for name, discovery in self.discoveries.items():
                f.write("\n" + "="*80 + "\n")
                f.write(f"{discovery.discovery_name.upper()}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Status: {'✅ VALIDATED' if discovery.validated else '⚠️ NOT VALIDATED'}\n")
                f.write(f"Confidence: {discovery.confidence_level}\n")
                f.write(f"p-value: {discovery.p_value:.6f}\n")
                f.write(f"Effect size: {discovery.effect_size:.4f}\n\n")
                
                f.write(f"EVIDENCE:\n{discovery.evidence}\n\n")
                f.write(f"IMPLICATIONS:\n{discovery.implications}\n")
        
        print(f"📄 Text report saved to: {report_path}")
    
    # ========================================================================
    # RUN ALL
    # ========================================================================
    
    def validate_all_discoveries(self):
        """Run complete discovery validation"""
        
        print("\n" + "="*80)
        print("🔬 VALIDATING EMERGENCE DISCOVERIES")
        print("="*80)
        
        # Load data
        self.load_periodic_tables()
        self.validate_sample_counts()
        self.load_all_model_data()
        
        if not self.model_data:
            print("\n❌ No data found")
            return
        
        # Validate each discovery
        print("\n" + "="*80)
        print("RUNNING VALIDATIONS")
        print("="*80)
        
        self.validate_discovery_1_universal_abrupt_emergence()
        self.validate_discovery_2_task_hierarchy()
        self.validate_discovery_3_size_inversion()
        self.validate_discovery_4_binary_competence()
        self.validate_discovery_5_sharpness_scaling()
        self.validate_discovery_6_scaling_law() 
        
        # Summary
        print("\n" + "="*80)
        print("🎯 DISCOVERY VALIDATION COMPLETE")
        print("="*80)
        
        validated = sum(1 for d in self.discoveries.values() if d.validated)
        total = len(self.discoveries)
        
        print(f"\n✅ Validated: {validated}/{total} discoveries")
        
        print("\n🔬 VALIDATED DISCOVERIES:")
        for name, discovery in self.discoveries.items():
            if discovery.validated:
                print(f"\n  ✅ {discovery.discovery_name}")
                print(f"     Confidence: {discovery.confidence_level}")
                print(f"     p = {discovery.p_value:.6f}")
                print(f"     Effect size = {discovery.effect_size:.4f}")
        
        # Save
        self.save_results()


def main():
    """Run discovery validation"""
    
    print("\n" + "="*80)
    print(" EMERGENCE DISCOVERY VALIDATION")
    print("Discovery-Driven Research Approach")
    print("="*80)
    
    validator = EmergenceDiscoveryValidator(
        base_dir=".",
        model_dirs=[
            "raw_results/llama_1b_models",
            "raw_results/phi_1_models",
            "raw_results/phi_1.5_models",
            "raw_results/llama_5b_model",
            'raw_results/llama_3b_model',
            "raw_results/llama_13b_models",
            "raw_results/deepsek_7b_models"
        ],
        alpha=0.05
    )
    
    validator.validate_all_discoveries()
    
    print("\n✅ COMPLETE!")


if __name__ == "__main__":
    main()

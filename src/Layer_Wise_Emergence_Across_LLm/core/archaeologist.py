
import re
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datasets import load_dataset
from tqdm import tqdm
from Layer_Wise_Emergence_Across_LLm.core.model_manager import SharedModelManager
from Layer_Wise_Emergence_Across_LLm.core.probing.layer_wise_prober import LayerWiseProber
from Layer_Wise_Emergence_Across_LLm.core.analysis.attention_analysis import EnhancedAttentionAnalyzer, AttentionAnalysisResult
from Layer_Wise_Emergence_Across_LLm.core.patching.activation_pather import ActivationPatcher

class EmergenceArchaeologist:
    """Main research framework orchestrating all experiments"""

    def __init__(self, output_dir: str = "./emergence_results", debug: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.encoding_vs_expression = {} 
        # Create subdirectories
        (self.output_dir / "probing").mkdir(exist_ok=True)
        (self.output_dir / "patching").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        self.current_model_manager = None
        print(f"Output directory: {self.output_dir}")



    def _get_or_load_model(self, model_name: str) -> SharedModelManager:
        """
        ✅ KEY FIX: Load model only if not already loaded
        """
        if self.current_model_manager is None or \
           self.current_model_manager.model_name != model_name:
            
            # Clean up old model if exists
            if self.current_model_manager is not None:
                print(f"Switching models: {self.current_model_manager.model_name} → {model_name}")
                self.current_model_manager.cleanup()
            
            # Load new model
            self.current_model_manager = SharedModelManager(
                model_name=model_name,
                debug=self.debug
            )
        else:
            print(f"Reusing already-loaded model: {model_name}")
        
        return self.current_model_manager


    def create_attention_visualizations(self, model_name: str):
        """
        Create publication-quality attention pattern visualizations.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load attention analysis
        attention_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_attention_analysis.json"

        if not attention_path.exists():
            print(f"No attention analysis found at {attention_path}")
            return

        with open(attention_path, 'r') as f:
            attention_data = json.load(f)

        layer_stats = attention_data.get('layer_stats', [])
        if not layer_stats:
            print("No layer stats in attention analysis")
            return

        # Calculate threshold dynamically based on model size
        num_layers = len(layer_stats)
        if num_layers <= 16:  # Small model (1B-3B)
            threshold = 0.15  # 5%
        elif num_layers < 30:  # Medium model (5B-7B)
            threshold = 0.03  # 10%
        else:  # Large model (13B+)
            threshold = 0.05  # 15%
        
        print(f"Visualization: Using threshold={threshold:.0%} for {num_layers}-layer model")

        # Extract data
        layers = [s['layer'] for s in layer_stats]
        math_attn = [s['avg_math_attention'] for s in layer_stats]
        reasoning_attn = [s['avg_reasoning_attention'] for s in layer_stats]
        specialized_heads = [s['num_consistent_specialized_heads'] for s in layer_stats]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Attention scores over layers
        ax = axes[0]
        ax.plot(layers, math_attn, 'o-', label='Math Attention', linewidth=2, markersize=6)
        ax.plot(layers, reasoning_attn, 's-', label='Reasoning Attention', linewidth=2, markersize=6)
        
        # Use dynamic threshold instead of hardcoded 0.15
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                  label=f'Threshold ({threshold:.0%})')

        # Mark emergence layer
        emergence_layer = attention_data.get('emergence_layer', -1)
        if emergence_layer >= 0:
            ax.axvline(x=emergence_layer, color='g', linestyle='--', alpha=0.7,
                      label=f'Emergence Layer {emergence_layer}')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.set_title(f'Attention Pattern Evolution - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Specialized heads per layer
        ax = axes[1]
        ax.bar(layers, specialized_heads, alpha=0.7, color='steelblue')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Number of Specialized Heads', fontsize=12)
        ax.set_title('Math-Specialized Attention Heads by Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        fig_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_attention_patterns.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f" Attention visualization saved to: {fig_path}")

        plt.close()


    def _analyze_attention_patterns(self, attention_results: List[List[AttentionAnalysisResult]],
                                   model_name: str) -> Dict:
        """
        Analyze aggregated attention patterns across all samples.

        Returns summary statistics.
        """
        if not attention_results:
            return {}

        # Use the analyzer's aggregation method
        analyzer = EnhancedAttentionAnalyzer(None, None)  # Dummy instance for aggregation
        aggregated = analyzer.aggregate_results(attention_results)

        # Add model-specific info
        aggregated['model_name'] = model_name
        aggregated['num_samples_analyzed'] = len(attention_results)

        return aggregated


    def load_dataset(self, dataset_name: str, split: str = None, num_samples: int = 50) -> List[Dict]:
        """Load various datasets for capability testing"""
        print(f"\n Loading {dataset_name} dataset...")

        samples = []

        if dataset_name == "gsm8k":
            # GSM8K only has 'train' and 'test' splits
            if split is None:
                split = "test"  # Default to test split
            dataset = load_dataset("gsm8k", "main", split=split)
            for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                question = item['question']
                answer_text = item['answer']
                match = re.search(r'####\s*(\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
                if match:
                    ground_truth = float(match.group(1).replace(',', ''))
                    samples.append({
                        'id': idx, 'question': question, 'ground_truth': ground_truth,
                        'dataset': dataset_name, 'task_type': 'math'
                    })

        elif dataset_name == "boolq":
            # BoolQ has 'train', 'validation', 'test' splits
            if split is None:
                split = "validation"  # Default to validation split
            dataset = load_dataset("boolq", split=split)
            for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                samples.append({
                    'id': idx,
                    'question': f"Passage: {item['passage']}\nQuestion: {item['question']}",
                    'ground_truth': item['answer'],  # True/False
                    'dataset': dataset_name,
                    'task_type': 'reasoning'
                })

        elif dataset_name == "commonsense_qa":
            # Commonsense QA has 'train', 'validation', 'test' splits
            if split is None:
                split = "validation"  # Default to validation split
            dataset = load_dataset("commonsense_qa", split=split)
            for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                question = f"{item['question']}\nOptions: " + " ".join([f"({chr(65+i)}) {opt}"
                                                                      for i, opt in enumerate(item['choices']['text'])])
                # Convert answer key to index
                answer_key = item['answerKey']
                if answer_key in ['A', 'B', 'C', 'D', 'E']:
                    answer_idx = ord(answer_key) - 65
                    ground_truth = item['choices']['text'][answer_idx]
                else:
                    continue

                samples.append({
                    'id': idx, 'question': question, 'ground_truth': ground_truth,
                    'dataset': dataset_name, 'task_type': 'commonsense'
                })

        elif dataset_name == "hellaswag":
            if split is None:
                split = "validation"
            
            try:
                dataset = load_dataset("Rowan/hellaswag", split=split)
                
                for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                    context = item['ctx']
                    endings = item['endings']
                    correct_idx = int(item['label'])
                    
                    # Store BOTH the number AND the text
                    # Use the LETTER (A, B, C, D) as ground truth for matching
                    ground_truth_letter = chr(65 + correct_idx)  # 0→A, 1→B, 2→C, 3→D
                    ground_truth_text = endings[correct_idx]
                    
                    # Format question with lettered options
                    question = f"Complete the following scenario:\n\n{context}\n\nWhich continuation makes the most sense?\n"
                    question += "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings)])
                    
                    samples.append({
                        'id': idx,
                        'question': question,
                        'ground_truth': ground_truth_letter,  # Use letter: "A", "B", "C", or "D"
                        'ground_truth_text': ground_truth_text,  # Store full text for reference
                        'dataset': dataset_name,
                        'task_type': 'commonsense'
                    })
                    
            except Exception as e:
                print(f"Error loading HellaSwag: {e}")
                return []


        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"Loaded {len(samples)} samples from {dataset_name} ({split} split)")
        return samples


    def analyze_enhanced_detection(self, df: pd.DataFrame, model_name: str):
        """Compare single-token vs multi-token detection patterns"""
        print(f"\nENHANCED DETECTION ANALYSIS")
        print("="*80)

        # Compare single-token vs multi-token by layer
        comparison_data = []
        for layer_idx in sorted(df['layer_idx'].unique()):
            layer_data = df[df['layer_idx'] == layer_idx]

            single_acc = layer_data['is_correct'].mean()
            multi_acc = layer_data['multi_token_correct'].mean()
            first_digit_acc = layer_data['first_digit_match'].mean()

            comparison_data.append({
              'layer': int(layer_idx),  # Convert to native int
              'single_token_accuracy': float(single_acc),  # Convert to float
              'multi_token_accuracy': float(multi_acc),
              'first_digit_accuracy': float(first_digit_acc),
              'improvement': float(multi_acc - single_acc)
          })

            # Print layer analysis
            improvement_str = f"(+{multi_acc-single_acc:.1%})" if multi_acc > single_acc else ""
            print(f"Layer {layer_idx:2d}: Single={single_acc:.1%} → Multi={multi_acc:.1%} {improvement_str}")

            # Flag major improvements
            if multi_acc > single_acc + 0.2:  # 20% improvement
                print(f" MAJOR IMPROVEMENT: Multi-token found what single-token missed!")

        # Find emergence layers
        single_emergence = next((l for l in comparison_data if l['single_token_accuracy'] > 0.5), None)
        multi_emergence = next((l for l in comparison_data if l['multi_token_accuracy'] > 0.5), None)

        print(f"\nEMERGENCE COMPARISON:")
        if single_emergence:
            print(f"  Single-token emergence: Layer {single_emergence['layer']} ({single_emergence['single_token_accuracy']:.1%})")
        if multi_emergence:
            print(f"  Multi-token emergence:  Layer {multi_emergence['layer']} ({multi_emergence['multi_token_accuracy']:.1%})")

        if multi_emergence and single_emergence and multi_emergence['layer'] < single_emergence['layer']:
            gap = single_emergence['layer'] - multi_emergence['layer']
            print(f"  Multi-token detects emergence {gap} layers EARLIER!")

        # Save enhanced analysis
        enhanced_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_enhanced_detection.json"
        with open(enhanced_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_compatible_data = json.loads(json.dumps(comparison_data, default=str))
            json.dump(json_compatible_data, f, indent=2)
        print(f"Enhanced detection analysis saved to: {enhanced_path}")

        return comparison_data


    def run_probing_experiment(self, model_name: str, samples: List[Dict],
                            task_type: str = "math") -> pd.DataFrame:
        """Run layer-wise probing on all samples"""
        print(f"\n{'='*80}")
        print(f"PROBING EXPERIMENT: {model_name}")
        print(f"{'='*80}\n")
        manager = self._get_or_load_model(model_name)
        prober = LayerWiseProber(manager, debug=self.debug)

        all_results = []
        successful_samples = 0

        for sample in tqdm(samples, desc="Probing samples"):
            try:
                layer_results = prober.probe_all_layers(
                    question=sample['question'],
                    ground_truth=sample['ground_truth'],
                    sample_id=sample['id'],
                    task_type=task_type,
                    dataset_name=sample['dataset']
                )

                # DEBUG: Check if we got any results
                if layer_results:
                    all_results.extend(layer_results)
                    successful_samples += 1
                    if self.debug:
                        print(f"Sample {sample['id']}: Got {len(layer_results)} layer results")
                else:
                    print(f"Sample {sample['id']}: No layer results returned")

            except Exception as e:
                print(f"Error on sample {sample['id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Probing completed: {successful_samples}/{len(samples)} samples successful")

        if not all_results:
            print("No probing results collected!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])
        prober.probing_results_df = df

        # DEBUG: Print DataFrame info
        print(f"Results DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")

        # SIMPLE ENCODING vs EXPRESSION ANALYSIS
        print(f"\n🔍 SIMPLE ENCODING vs EXPRESSION ANALYSIS:")
        try:
            encoding_vs_expression = prober.linear_prober.compare_encoding_vs_expression_v2(df)

            # CRITICAL: Store for use in analyze_emergence_patterns
            self.encoding_vs_expression = encoding_vs_expression

            early_encoding_detected = False
            for layer_idx, results in encoding_vs_expression.items():
                expr_acc = results['expression_accuracy']
                enc_acc = results['encoding_accuracy']
                gap = results['gap']

                if enc_acc is not None:
                    print(f"   Layer {layer_idx:2d}: Expression={expr_acc:.1%}, Encoding={enc_acc:.1%}, Gap={gap:+.1%}")

                    # Check for early encoding detection
                    if layer_idx < prober.num_layers // 2 and enc_acc > expr_acc + 0.3:
                        early_encoding_detected = True
                        print(f"   EARLY ENCODING DETECTED at layer {layer_idx}!")

            if early_encoding_detected:
                print(f"\n CRITICAL FINDING: Capabilities are ENCODED in middle layers but not EXPRESSED!")

            # Save encoding analysis
            encoding_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_encoding_analysis.json"
            with open(encoding_path, 'w') as f:
                json.dump(encoding_vs_expression, f, indent=2)
            print(f" Encoding vs Expression analysis saved to: {encoding_path}")

        except Exception as e:
            print(f" Encoding vs Expression analysis failed: {e}")

        # ADD MULTI-TOKEN ANALYSIS
        print(f"\n ANALYZING MULTI-TOKEN PROBING RESULTS...")
        multi_token_analysis = self._analyze_multi_token_results(df)

        # ATTENTION ANALYSIS
        print(f"\n🔍 ANALYZING ATTENTION PATTERNS...")
        try:
            if hasattr(prober, '_attention_results') and prober._attention_results:
                attention_analysis = self._analyze_attention_patterns(
                    prober._attention_results, model_name
                )

                # Save attention analysis
                attention_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_attention_analysis.json"
                with open(attention_path, 'w') as f:
                    json.dump(attention_analysis, f, indent=2)
                print(f" Attention analysis saved to: {attention_path}")

                # Print key findings
                if attention_analysis:
                    print(f"\n ATTENTION FINDINGS:")
                    print(f"   Attention emergence layer: {attention_analysis.get('emergence_layer', 'N/A')}")

                    # Find layer with most specialized heads
                    layer_stats = attention_analysis.get('layer_stats', [])
                    if layer_stats:
                        max_heads_layer = max(layer_stats,
                                            key=lambda x: x['num_consistent_specialized_heads'])
                        print(f"   Peak specialized heads: Layer {max_heads_layer['layer']} "
                              f"({max_heads_layer['num_consistent_specialized_heads']} heads)")
                        print(f"   Math attention at peak: {max_heads_layer['avg_math_attention']:.1%}")
            else:
                print("   No attention results available")

        except Exception as e:
            print(f"   Attention analysis failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

        if multi_token_analysis:
            # print(f"   MULTI-TOKEN FINDINGS:")
            # print(f"   Samples where multi-token found capabilities earlier: {multi_token_analysis['early_detection_samples']}")
            # print(f"   Layers with multi-token advantage: {multi_token_analysis['advantage_layers']}")
            # print(f"   First digit match success rate: {multi_token_analysis['first_digit_success']:.1%}")

            # Save multi-token analysis
            multi_token_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_multi_token_analysis.json"
            with open(multi_token_path, 'w') as f:
                json.dump(multi_token_analysis, f, indent=2)
            print(f" Multi-token analysis saved to: {multi_token_path}")


        # #  ADD ENHANCED DETECTION ANALYSIS
        # print(f"\n COMPREHENSIVE DETECTION COMPARISON...")
        # enhanced_comparison = self.analyze_enhanced_detection(df, model_name)

        # # Check for critical findings
        # early_multi_detection = any(
        #     layer['multi_token_accuracy'] > layer['single_token_accuracy'] + 0.3
        #     and layer['layer'] < prober.num_layers // 2
        #     for layer in enhanced_comparison
        # )

        # if early_multi_detection:
        #     print(f"\n CRITICAL FINDING: Multi-token probing reveals capabilities in EARLY layers!")
        #     print(f"   Your single-token method was missing early-stage capability formation!")

        # Save results
        model_safe = model_name.replace('/', '_')
        output_path = self.output_dir / "probing" / f"{model_safe}_probing.csv"
        df.to_csv(output_path, index=False)
        print(f"\n Probing results saved to: {output_path}")

        return df

    def _analyze_multi_token_results(self, df: pd.DataFrame) -> dict:
        """Analyze multi-token probing results vs single-token"""
        if 'multi_token_correct' not in df.columns:
            return {}

        analysis = {
            'early_detection_samples': 0,
            'advantage_layers': [],
            'first_digit_success': 0.0,
            'layer_breakdown': {}
        }

        # Analyze by layer
        for layer_idx in sorted(df['layer_idx'].unique()):
            layer_data = df[df['layer_idx'] == layer_idx]

            single_token_acc = layer_data['is_correct'].mean()
            multi_token_acc = layer_data['multi_token_correct'].mean()
            first_digit_acc = layer_data['first_digit_match'].mean()

            analysis['layer_breakdown'][int(layer_idx)] = {  # Convert key to native int
                'single_token_accuracy': float(single_token_acc),  # Convert to float
                'multi_token_accuracy': float(multi_token_acc),
                'first_digit_accuracy': float(first_digit_acc),
                'multi_token_advantage': float(multi_token_acc - single_token_acc)
            }

            # Count layers where multi-token has advantage
            # Count layers where multi-token has advantage
            if multi_token_acc > single_token_acc + 0.1:  # 10% advantage
                analysis['advantage_layers'].append(int(layer_idx))  # Convert to native int

        # Count samples where multi-token detected capabilities earlier
        sample_analysis = df.groupby('sample_id').apply(
            lambda x: self._find_earliest_detection(x)
        )
        analysis['early_detection_samples'] = int(sample_analysis.sum())  # Convert to native int
        # First digit success rate
        analysis['first_digit_success'] = df['first_digit_match'].mean()

        return analysis

    def _find_earliest_detection(self, sample_data: pd.DataFrame) -> bool:
        """Check if multi-token detected capability earlier than single-token for a sample"""
        single_token_layers = sample_data[sample_data['is_correct'] == True]['layer_idx']
        multi_token_layers = sample_data[sample_data['multi_token_correct'] == True]['layer_idx']

        if len(multi_token_layers) == 0:
            return False

        if len(single_token_layers) == 0:
            return True  # Multi-token found it when single-token didn't

        earliest_single = min(single_token_layers)
        earliest_multi = min(multi_token_layers)

        return earliest_multi < earliest_single

    def run_patching_experiment(self, model_name: str, samples: List[Dict],
                                layers_to_test: Optional[List[int]] = None,
                                intervention_type: str = "zero_ablate") -> pd.DataFrame:
        """Run activation patching experiments"""
        print(f"\n{'='*80}")
        print(f"🔧 PATCHING EXPERIMENT: {model_name}")
        print(f"{'='*80}\n")


        manager = self._get_or_load_model(model_name)

        patcher = ActivationPatcher(manager, debug=self.debug)

        if layers_to_test is None:
            # Test every 4th layer for efficiency
            layers_to_test = list(range(0, patcher.num_layers, 4))

        print(f"Testing {len(layers_to_test)} layers: {layers_to_test}")

        all_results = []
        for sample in tqdm(samples[:10], desc="Patching samples"):  # Limit samples for patching
            for layer_idx in layers_to_test:
                try:
                    result = patcher.patch_layer_activation(
                        question=sample['question'],
                        ground_truth=sample['ground_truth'],
                        sample_id=sample['id'],
                        layer_to_patch=layer_idx,
                        intervention_type=intervention_type,
                        dataset_name=sample['dataset']
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f" Error patching layer {layer_idx} on sample {sample['id']}: {e}")
                    continue

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])

        # Save results
        model_safe = model_name.replace('/', '_')
        output_path = self.output_dir / "patching" / f"{model_safe}_patching_{intervention_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n Patching results saved to: {output_path}")

        return df

    def analyze_emergence_patterns(self, probing_df: pd.DataFrame) -> Dict:
        """Analyze where and how capabilities emerge across layers"""
        print(f"\n{'='*80}")
        print(f"📊 EMERGENCE ANALYSIS")
        print(f"{'='*80}\n")
    
        print(f"📋 DataFrame columns: {list(probing_df.columns)}")
        print(f"📊 DataFrame shape: {probing_df.shape}")
    
        if probing_df.empty:
            print(" Empty DataFrame - skipping analysis")
            return {}
    
        required_columns = ['layer_idx', 'is_correct', 'layer_confidence']
        missing_columns = [col for col in required_columns if col not in probing_df.columns]
    
        if missing_columns:
            print(f" Missing required columns: {missing_columns}")
            print(f" Available columns: {list(probing_df.columns)}")
            return {}
    
        analysis = {}
    
        # 1. Layer-wise accuracy
        layer_stats = probing_df.groupby('layer_idx').agg({
            'is_correct': 'mean',
            'layer_confidence': 'mean',
            'activation_mean': 'mean',
            'activation_std': 'mean',
            'activation_sparsity': 'mean'
        }).reset_index()
    
        layer_stats.columns = ['layer', 'accuracy', 'confidence', 'act_mean', 'act_std', 'sparsity']
    
        print("Layer-wise Statistics:")
        print(layer_stats.round(4).to_string())
    
        # 2. Check for encoding vs expression results from linear probes
        encoding_emergence_layer = -1
        expression_emergence_layer = -1
        
        if hasattr(self, 'encoding_vs_expression') and self.encoding_vs_expression:
            print(f"\n DETECTING EMERGENCE FROM LINEAR PROBES:")
            
            for layer_idx in sorted(self.encoding_vs_expression.keys()):
                enc_acc = self.encoding_vs_expression[layer_idx].get('encoding_accuracy')
                expr_acc = self.encoding_vs_expression[layer_idx].get('expression_accuracy')
                
                # Emergence = first layer with >70% encoding accuracy
                if enc_acc is not None and enc_acc > 0.7 and encoding_emergence_layer == -1:
                    encoding_emergence_layer = layer_idx
                    print(f"    ENCODING emergence at layer {layer_idx} ({enc_acc:.1%} linear probe accuracy)")
                
                # Expression emergence = first layer with >70% expression accuracy  
                if expr_acc is not None and expr_acc > 0.7 and expression_emergence_layer == -1:
                    expression_emergence_layer = layer_idx
                    print(f"    EXPRESSION emergence at layer {layer_idx} ({expr_acc:.1%} next-token accuracy)")
            
            if encoding_emergence_layer > 0 and expression_emergence_layer > 0:
                gap = expression_emergence_layer - encoding_emergence_layer
                print(f"    GAP: {gap} layers between where capability is computed vs expressed")
            elif encoding_emergence_layer > 0:
                print(f"    Capability is ENCODED at layer {encoding_emergence_layer} but never strongly EXPRESSED")
    
        # 3. Crystallization layer detection (improved)
        print(f"\n DEBUG: Detailed layer accuracy progression:")
        for layer in range(len(layer_stats)):
            layer_data = probing_df[probing_df['layer_idx'] == layer]
            if len(layer_data) > 0:
                acc = layer_data['is_correct'].mean()
                conf = layer_data['layer_confidence'].mean()
                sample_correctness = [f"{sample_id}:{correct}" for sample_id, correct in
                                    zip(layer_data['sample_id'], layer_data['is_correct'])]
                print(f"Layer {layer:2d}: accuracy={acc:.3f}, confidence={conf:.3f}, samples={sample_correctness}")
    
        crystallization_layer = -1
        detection_method = "not_found"
    
        # More lenient thresholds
        sustained_threshold = 0.7  # 70% accuracy (was 80%)
        min_sustained_layers = 3
        confidence_requirement = 0.01
    
        num_layers = len(layer_stats)
        skip_threshold = max(3, num_layers // 10)  # Skip only first 10%
    
        print(f" Crystallization search: Skipping layers 0-{skip_threshold-1}")
    
        # Strategy 1: Sustained high accuracy
        for i in range(skip_threshold, len(layer_stats) - min_sustained_layers + 1):
            window = layer_stats.iloc[i:i + min_sustained_layers]
            window_confidences = layer_stats['confidence'].iloc[i:i + min_sustained_layers]
    
            if (all(window['accuracy'] >= sustained_threshold) and
                all(window_confidences >= confidence_requirement)):
                crystallization_layer = i
                detection_method = 'sustained_high_accuracy'
                print(f" Found crystallization at layer {i} via sustained accuracy")
                break
    
        # Strategy 2: Significant jump
        if crystallization_layer == -1:
            accuracy_jump_threshold = 0.3
            for i in range(skip_threshold, len(layer_stats)):
                if i == 0:
                    continue
                prev_acc = layer_stats['accuracy'].iloc[i-1]
                curr_acc = layer_stats['accuracy'].iloc[i]
                curr_conf = layer_stats['confidence'].iloc[i]
    
                if (curr_acc - prev_acc >= accuracy_jump_threshold and
                    curr_conf >= 0.01 and curr_acc >= 0.5):
                    crystallization_layer = i
                    detection_method = 'significant_jump'
                    print(f" Found crystallization at layer {i} via significant jump")
                    break
    
        # Strategy 3: Fallback
        if crystallization_layer == -1:
            late_layers_stats = layer_stats.iloc[skip_threshold:]
            if len(late_layers_stats) > 0:
                max_acc_idx = late_layers_stats['accuracy'].idxmax()
                crystallization_layer = max_acc_idx
                detection_method = 'max_accuracy_fallback'
                print(f" Found crystallization at layer {crystallization_layer} via fallback")
    
        # Validate crystallization
        if crystallization_layer >= 0:
            crystal_acc = layer_stats['accuracy'].iloc[crystallization_layer]
            crystal_conf = layer_stats['confidence'].iloc[crystallization_layer]
            print(f" Validated crystallization: Layer {crystallization_layer}, "
                  f"Accuracy={crystal_acc:.3f}, Confidence={crystal_conf:.3f}")
    
        # 4. Calculate capability evolution
        early_layers = probing_df[probing_df['layer_idx'] < num_layers // 4]
        mid_layers = probing_df[(probing_df['layer_idx'] >= num_layers // 4) &
                               (probing_df['layer_idx'] < num_layers // 2)]
        late_layers = probing_df[probing_df['layer_idx'] >= num_layers // 2]
    
        print(f"\n Layer grouping: Early (0-{num_layers//4-1}), "
              f"Mid ({num_layers//4}-{num_layers//2-1}), Late ({num_layers//2}-{num_layers-1})")
    
        # Store results
        analysis['layer_statistics'] = layer_stats.to_dict('records')
        analysis['crystallization_layer'] = int(crystallization_layer)
        analysis['crystallization_method'] = detection_method
        analysis['encoding_emergence_layer'] = int(encoding_emergence_layer) if encoding_emergence_layer > 0 else None
        analysis['expression_emergence_layer'] = int(expression_emergence_layer) if expression_emergence_layer > 0 else None
        analysis['num_layers'] = int(probing_df['layer_idx'].max() + 1)
        analysis['overall_accuracy'] = float(layer_stats['accuracy'].iloc[-1])
    
        analysis['capability_evolution'] = {
            'early_accuracy': float(early_layers['is_correct'].mean()) if len(early_layers) > 0 else 0.0,
            'mid_accuracy': float(mid_layers['is_correct'].mean()) if len(mid_layers) > 0 else 0.0,
            'late_accuracy': float(late_layers['is_correct'].mean()) if len(late_layers) > 0 else 0.0,
            'early_confidence': float(early_layers['layer_confidence'].mean()) if len(early_layers) > 0 else 0.0,
            'late_confidence': float(late_layers['layer_confidence'].mean()) if len(late_layers) > 0 else 0.0,
        }
    
        analysis['activation_evolution'] = {
            'early_sparsity': float(early_layers['activation_sparsity'].mean()) if len(early_layers) > 0 else 0.0,
            'late_sparsity': float(late_layers['activation_sparsity'].mean()) if len(late_layers) > 0 else 0.0,
            'activation_growth': float(late_layers['activation_std'].mean()) /
                               float(early_layers['activation_std'].mean()) if len(early_layers) > 0 else 1.0
        }
    
        # Warnings
        print(f"\n KEY FINDINGS:")
        
        if crystallization_layer >= num_layers - 2:
            print(f"   WARNING: Crystallization at layer {crystallization_layer}/{num_layers} (final layer!)")
            print(f"     This suggests detection is only catching the OUTPUT, not computation")
            if encoding_emergence_layer > 0:
                print(f"     TRUE emergence is likely at layer {encoding_emergence_layer} (from linear probes)")
        
        if encoding_emergence_layer > 0:
            print(f"  - Encoding emergence: Layer {encoding_emergence_layer} ({encoding_emergence_layer/num_layers*100:.0f}% depth)")
        if expression_emergence_layer > 0:
            print(f"  - Expression emergence: Layer {expression_emergence_layer} ({expression_emergence_layer/num_layers*100:.0f}% depth)")
        print(f"  - Crystallization layer: {crystallization_layer} (method: {detection_method})")
        print(f"  - Accuracy progression: {analysis['capability_evolution']['early_accuracy']:.1%} → "
              f"{analysis['capability_evolution']['mid_accuracy']:.1%} → "
              f"{analysis['capability_evolution']['late_accuracy']:.1%}")
        print(f"  - Confidence growth: {analysis['capability_evolution']['early_confidence']:.3f} → "
              f"{analysis['capability_evolution']['late_confidence']:.3f}")
        print(f"  - Final accuracy: {analysis['overall_accuracy']:.2%}")
        print(f"  - Activation sparsity: {analysis['activation_evolution']['early_sparsity']:.1%} → "
              f"{analysis['activation_evolution']['late_sparsity']:.1%}")
    
        # Save analysis
        analysis_path = self.output_dir / "analysis" / "emergence_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n Analysis saved to: {analysis_path}")
    
        return analysis

    def cleanup_models(self):
        """
        Clean up models when completely done
        """
        if self.current_model_manager is not None:
            self.current_model_manager.cleanup()
            self.current_model_manager = None


    def create_periodic_table(self, multi_model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a 'periodic table' of capability emergence across models and tasks
        """
        print(f"\n{'='*80}")
        print(f"CREATING PERIODIC TABLE OF EMERGENCE")
        print(f"{'='*80}\n")

        periodic_data = []

        for model_name, probing_df in multi_model_results.items():
            print(f"Analyzing {model_name}...")

            # Skip if DataFrame is empty
            if probing_df.empty:
                print(f" Skipping {model_name} - empty DataFrame")
                continue

            analysis = self.analyze_emergence_patterns(probing_df)

            # Skip if analysis failed
            if not analysis:
                print(f" Skipping {model_name} - analysis failed")
                continue

            periodic_data.append({
                'model': model_name,
                'num_layers': analysis['num_layers'],
                'crystallization_layer': analysis['crystallization_layer'],
                'crystallization_depth_pct': (analysis['crystallization_layer'] / analysis['num_layers'] * 100)
                                              if analysis['crystallization_layer'] > 0 else 0,
                'final_accuracy': analysis['overall_accuracy'],
                'early_sparsity': analysis['activation_evolution']['early_sparsity'],
                'late_sparsity': analysis['activation_evolution']['late_sparsity']
            })

        if not periodic_data:
            print(" No valid data for periodic table")
            return pd.DataFrame()

        periodic_table = pd.DataFrame(periodic_data)

        print("\n PERIODIC TABLE OF CAPABILITY EMERGENCE:")
        print("="*80)
        print(periodic_table.to_string(index=False))
        print("="*80)

        # Save
        table_path = self.output_dir / "analysis" / "periodic_table.csv"
        periodic_table.to_csv(table_path, index=False)
        print(f"\n Periodic table saved to: {table_path}")

        return periodic_table
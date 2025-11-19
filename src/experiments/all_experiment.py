from Layer_Wise_Emergence_Across_LLm.core.archaeologist import EmergenceArchaeologist
from Layer_Wise_Emergence_Across_LLm.core.utils.calculate_path  import calculate_patch_layers
import pandas as pd
import json


def main():
    """Main research pipeline - FIXED TO ACCUMULATE ALL DATASETS"""
    from huggingface_hub import login
    login(token="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")
    
    MODELS_TO_TEST = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct", 
        "prithivMLmods/Llama-3.1-5B-Instruct",
        "meta-llama/Llama-2-13b-hf"
        "google/gemma-2-2b-it",
        "microsoft/phi-1_5",
        "microsoft/phi-1",
        "deepseek-ai/deepseek-llm-7b-base"
    ]

    DATASETS_TO_TEST = [
        ("gsm8k", "math", "test"),
        ("boolq", "reasoning", "validation"),
        ("commonsense_qa", "commonsense", "validation"),
        ("hellaswag", "commonsense", "validation"),      
    ]
    
    NUM_SAMPLES = 400
    DEBUG = False
    
    archaeologist = EmergenceArchaeologist(
        output_dir="./emergence_archaeology_results",
        debug=DEBUG
    )
    
    all_results = {}
    
    for model_name in MODELS_TO_TEST:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING MODEL: {model_name}")
        print(f"{'#'*80}\n")
        
        all_probing_dfs = []
        
        for dataset_name, task_type, split in DATASETS_TO_TEST:
            print(f"\n--- Testing {model_name} on {dataset_name} ---")
            
            samples = archaeologist.load_dataset(
                dataset_name=dataset_name,
                split=split,
                num_samples=NUM_SAMPLES
            )
            
            try:
                probing_df = archaeologist.run_probing_experiment(
                    model_name=model_name,
                    samples=samples,
                    task_type=task_type
                )
                
                probing_df['dataset_name'] = dataset_name
                probing_df['task_type'] = task_type
                
                all_probing_dfs.append(probing_df)
                
                print(f" Completed {dataset_name}: {len(probing_df)} rows")
                
            except Exception as e:
                print(f" Error on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_probing_dfs:
            combined_df = pd.concat(all_probing_dfs, ignore_index=True)
            
            print(f"\n COMBINED RESULTS FOR {model_name}:")
            print(f"   Total samples: {combined_df['sample_id'].nunique()}")
            print(f"   Total rows: {len(combined_df)}")
            print(f"   Datasets: {combined_df['dataset_name'].unique()}")
            
            model_safe = model_name.replace('/', '_')
            output_path = archaeologist.output_dir / "probing" / f"{model_safe}_probing_combined.csv"
            combined_df.to_csv(output_path, index=False)
            print(f" Saved combined data to: {output_path}")
            
            analysis = archaeologist.analyze_emergence_patterns(combined_df)
            
            all_results[f"{model_name}_combined"] = combined_df
            
            print(f"\n Per-Dataset Emergence:")
            for dataset_name in combined_df['dataset_name'].unique():
                dataset_df = combined_df[combined_df['dataset_name'] == dataset_name]
                dataset_analysis = archaeologist.analyze_emergence_patterns(dataset_df)
                
                if dataset_analysis:
                    print(f"   {dataset_name}: Layer {dataset_analysis['crystallization_layer']} "
                          f"({dataset_analysis['crystallization_layer']/dataset_analysis['num_layers']*100:.1f}%)")
                    
                    dataset_analysis_path = archaeologist.output_dir / "analysis" / f"{model_safe}_{dataset_name}_emergence.json"
                    with open(dataset_analysis_path, 'w') as f:
                        json.dump(dataset_analysis, f, indent=2)
            
            if len(combined_df) >= 30:  
                try:
                    model_num_layers = len(archaeologist.current_model_manager.model.model.layers)
                    auto_patch_layers = calculate_patch_layers(
                        num_layers=model_num_layers,
                        num_patches=5
                    )
                    
                    patch_samples = []
                    for dataset_name in combined_df['dataset_name'].unique()[:2]:  
                        dataset_samples = combined_df[combined_df['dataset_name'] == dataset_name].head(5)
                        for _, row in dataset_samples.iterrows():
                            patch_samples.append({
                                'id': row['sample_id'],
                                'question': row['question'],
                                'ground_truth': row['ground_truth'],
                                'dataset': row['dataset_name']
                            })
                    
                    if patch_samples:
                        patching_df = archaeologist.run_patching_experiment(
                            model_name=model_name,
                            samples=patch_samples,
                            layers_to_test=auto_patch_layers,  
                            intervention_type="residual_patch"
                        )
                except Exception as e:
                    print(f" Patching failed: {e}")
        
        
    
    archaeologist.cleanup_models()
    
    if all_results:
        archaeologist.create_periodic_table(all_results)
    
    print(f"\n CREATING ATTENTION VISUALIZATIONS...")
    for model_name in MODELS_TO_TEST:
        try:
            archaeologist.create_attention_visualizations(model_name)
        except Exception as e:
            print(f" Failed to create attention viz for {model_name}: {e}")

    print(f"\n\n{'#'*80}")
    print(f"# EMERGENCE ARCHAEOLOGY COMPLETE")
    print(f"# Results saved to: {archaeologist.output_dir}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
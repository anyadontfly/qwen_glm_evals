import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot online benchmark results comparing Qwen and GLM models.")
    parser.add_argument("suffix", type=str, help="Directory suffix (e.g., '_burst_sweep_random')")
    parser.add_argument("x_var", type=str, help="X-axis variable name (e.g., 'burstiness')")
    parser.add_argument("--output-dir", type=str, default="plots", help="Base output directory for plots")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "results")
    
    # Create output directory based on x_var
    if os.path.isabs(args.output_dir):
        base_out = args.output_dir
    else:
        base_out = os.path.join(script_dir, args.output_dir)
        
    plots_dir = os.path.join(base_out, args.x_var)
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Define the two models to compare
    models = {
        "Qwen": "Qwen-Qwen3.5-9B",
        "GLM": "zai-org-GLM-4.7-Flash"
    }
    
    # Load dataframes
    data = {}
    
    # Store standard metrics we care about
    metrics_to_plot = [
        'output_throughput', 
        'request_throughput', 
        'mean_ttft_ms', 
        'mean_tpot_ms', 
        'mean_itl_ms', 
        'mean_e2el_ms',
        'p99_ttft_ms', 
        'p99_tpot_ms', 
        'p99_itl_ms', 
        'p99_e2el_ms'
    ]

    for label, base_name in models.items():
        dir_name = f"{base_name}{args.suffix}"
        full_path = os.path.join(results_base_dir, dir_name, "summary.csv")
        
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                # Verify x_var exists
                if args.x_var in df.columns:
                    # Sort by x_var
                    df = df.sort_values(by=args.x_var)
                    data[label] = df
                    print(f"Loaded data for {label} from {dir_name}")
                else:
                    print(f"Warning: Column '{args.x_var}' not found in {full_path}. Available columns: {list(df.columns)}")
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
        else:
            print(f"Warning: Directory/File not found: {full_path}")

    if not data:
        print("No data found to plot.")
        sys.exit(1)

    # Create plots for each metric
    for metric in metrics_to_plot:
        has_any_data = False
        plt.figure(figsize=(10, 6))
        
        colors = {'Qwen': '#1f77b4', 'GLM': '#ff7f0e'} 
        
        for label, df in data.items():
            if metric in df.columns:
                # Filter out inf or nan if any, for safety
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.x_var, metric])
                if not clean_df.empty:
                    plt.plot(clean_df[args.x_var], clean_df[metric], marker='o', label=label, color=colors.get(label))
                    has_any_data = True
        
        if not has_any_data:
            plt.close()
            continue

        plt.xlabel(args.x_var)
        plt.ylabel(metric)
        plt.title(f"{args.x_var} vs {metric}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        output_filename = f"combined{args.suffix}_{metric}.png"
        output_path = os.path.join(plots_dir, output_filename)
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        plt.close()

if __name__ == "__main__":
    main()

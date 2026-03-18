import os
import matplotlib.pyplot as plt
import re
import glob
import sys

# Define base directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results_h100_cu126")

# Verify base directory exists
if not os.path.exists(base_dir):
    print(f"Error: Base directory containing results not found at {base_dir}")
    sys.exit(1)

models = {
    "glm-4.7-flash": "GLM-4.7-Flash", 
    "qwen-9b": "Qwen3.5-9B",
    "qwen-27b": "Qwen3.5-27B"
}
# Note: Folder names are keys, Plot labels are values. 
# Adjust labels based on file contents if needed, e.g. Qwen3.5 observed in file content.

accuracies = {}

print(f"Scanning results in {base_dir}...")

for model_dir, model_name in models.items():
    summary_path = os.path.join(base_dir, model_dir, "summary")
    
    if not os.path.exists(summary_path):
        print(f"Warning: Summary path not found for {model_dir}: {summary_path}")
        continue
    
    # Find all summary files
    files = glob.glob(os.path.join(summary_path, "*_summary.txt"))
    if not files:
        print(f"Warning: No summary file found for {model_dir} in {summary_path}")
        continue
        
    # Pick the latest file based on modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"Parsing {latest_file} for {model_name}...")
    
    try:
        with open(latest_file, "r") as f:
            content = f.read()
            # Look for the final average accuracy line
            # Format: "Average accuracy: 0.7704"
            match = re.search(r"Average accuracy:\s*([0-9.]+)", content)
            if match:
                acc = float(match.group(1))
                accuracies[model_name] = acc
                print(f"  Found accuracy: {acc}")
            else:
                print(f"  Warning: Could not parse 'Average accuracy:' from {os.path.basename(latest_file)}")
    except Exception as e:
        print(f"  Error reading file {latest_file}: {e}")

# Plotting
if accuracies:
    plt.figure(figsize=(10, 6))
    
    # Sort or keep order? Let's keep the order defined in 'models' if found
    plot_names = []
    plot_values = []
    
    for key, name in models.items():
        if name in accuracies:
            plot_names.append(name)
            plot_values.append(accuracies[name])
            
    bars = plt.bar(plot_names, plot_values, color=['#4daf4a', '#377eb8', '#e41a1c'])
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.title('MMLU-Pro Average Accuracy Comparison', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10)
    
    output_file = os.path.join(script_dir, 'mmlu_pro_accuracy_comparison.png')
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
else:
    print("\nNo accuracy data collected to plot.")

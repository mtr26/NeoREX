import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_benchmarks():
    models = ["rex", "neorex"]
    optimizers = ["adam", "muon"]
    steps = 1500 # 1500 steps * 4096 tokens/step = ~6.1M tokens
    
    # 50M tokens with batch size 4 and seq len 1024 is ~12,207 steps.
    # For a quick local benchmark run, we default to 50 steps.
    
    results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_paths = {}

    for model in models:
        for opt in optimizers:
            name = f"{model}_{opt}"
            print(f"========== Running Benchmark: {name} ==========")
            csv_path = os.path.join(results_dir, f"{name}.csv")
            if os.path.exists(csv_path):
                os.remove(csv_path)
                
            cmd = [
                "python", "training/benchmark.py",
                "--model", model,
                "--optimizer", opt,
                "--csv_out", csv_path,
                "--steps", str(steps)
            ]
            
            # Subprocess ensures CUDA memory is completely cleared between runs
            subprocess.run(cmd, check=True)
            csv_paths[name] = csv_path
            
    plot_results(csv_paths, results_dir)

def plot_results(csv_paths, results_dir):
    plt.figure(figsize=(15, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    for name, path in csv_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Smooth the loss
            smoothed = df['loss'].rolling(window=5, min_periods=1).mean()
            plt.plot(df['step'], smoothed, label=name)
    plt.title("Convergence (Loss over Steps)")
    plt.xlabel("Step")
    plt.ylabel("Smoothed Loss")
    plt.legend()
    
    # Extract Peak VRAM and Throughput averages
    names = []
    vram = []
    throughput = []
    
    for name, path in csv_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            names.append(name)
            vram.append(df['vram_mb'].max())
            throughput.append(df['tokens_per_sec'].mean())
            
    # 2. Peak VRAM
    plt.subplot(1, 3, 2)
    bars = plt.bar(names, vram, color=['blue', 'orange', 'green', 'red'])
    plt.title("Peak VRAM Footprint")
    plt.ylabel("MB")
    plt.xticks(rotation=45)
    
    # 3. Throughput
    plt.subplot(1, 3, 3)
    bars = plt.bar(names, throughput, color=['blue', 'orange', 'green', 'red'])
    plt.title("Throughput")
    plt.ylabel("Tokens / Sec")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "benchmark_plots.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    run_benchmarks()

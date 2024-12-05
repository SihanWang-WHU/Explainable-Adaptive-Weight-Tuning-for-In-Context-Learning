import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load all evaluation result CSV files from a directory
def load_eval_csv_files(directory_path):
    file_paths = glob.glob(f"{directory_path}/*_eval_loss.csv")
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df["source"] = file_path.split("/")[-1]
        dataframes.append(df)
    return dataframes

# Plot loss curves for evaluation
def plot_eval_loss_curves(dataframes):
    plt.figure(figsize=(10, 6))
    for df in dataframes:
        plt.plot(
            df["epoch"],
            df["loss"],
            label=df["source"].iloc[0],
            # s=10,  # Size of the points
            # alpha=0.7  # Transparency for readability
        )
    plt.title("Evaluation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot EM and F1 curves for evaluation
def plot_eval_metrics_curves(dataframes):
    plt.figure(figsize=(10, 6))
    for df in dataframes:
        plt.plot(
            df["epoch"],
            df["EM"],
            label=f"EM - {df['source'].iloc[0]}",
            marker='o',
            linestyle='--',
            alpha=0.7
        )
        plt.plot(
            df["epoch"],
            df["F1"],
            label=f"F1 - {df['source'].iloc[0]}",
            marker='s',
            linestyle='--',
            alpha=0.7
        )
    plt.title("Evaluation Metrics (EM and F1) Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
directory_path = "../results"  # Replace with your directory path
eval_dataframes = load_eval_csv_files(directory_path)
plot_eval_loss_curves(eval_dataframes)
plot_eval_metrics_curves(eval_dataframes)

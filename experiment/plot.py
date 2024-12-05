import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load all CSV files from a directory
def load_csv_files(directory_path):
    file_paths = glob.glob(f"{directory_path}/*_train_loss.csv")
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df["source"] = file_path.split("/")[-1]
        dataframes.append(df)
    return dataframes

# Plot loss curves
def plot_loss_curves(dataframes):
    plt.figure(figsize=(10, 6))
    for df in dataframes[:100]:
        df = df[:500]
        plt.scatter(
            df["epoch"] + df["batch"] / df["batch"].max(),
            df["loss"],
            label=df["source"].iloc[0],
            s=1,  # Size of the points
            alpha=0.7  # Transparency for readability
        )
    plt.title("Loss Curves")
    plt.xlabel("Epoch + Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot learning rate curves
def plot_lr_curves(dataframes):
    plt.figure(figsize=(10, 6))
    for df in dataframes:
        plt.scatter(
            df["epoch"] + df["batch"] / df["batch"].max(),
            df["lr"],
            label=df["source"].iloc[0],
            s=3,  # Size of the points
            alpha=0.7  # Transparency for readability
        )

    plt.title("Learning Rate Curves")
    plt.xlabel("Epoch + Batch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
directory_path = "../results"  # Replace with your directory path
dataframes = load_csv_files(directory_path)
plot_loss_curves(dataframes)
plot_lr_curves(dataframes)

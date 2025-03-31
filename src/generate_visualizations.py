import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(csv_filename, save_folder):
    """
    Plot histograms for each mask coefficient.
    """
    if not os.path.exists(csv_filename):
        print(f"CSV file {csv_filename} not found!")
        return

    # Read the CSV file.
    df = pd.read_csv(csv_filename)
    # Identify all columns that correspond to mask coefficients (mask_1, mask_2, ..., mask_32).
    mask_columns = [col for col in df.columns if col.startswith("mask_")]
    num_masks = len(mask_columns)

    # Determine grid layout: use 4 columns (rows = ceil(num_masks/4)).
    cols = 4
    rows = num_masks // cols if num_masks % cols == 0 else (num_masks // cols) + 1

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()

    # Plot each histogram.
    for i, mask in enumerate(mask_columns):
        axs[i].hist(df[mask].dropna(), bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(mask)
        axs[i].set_xlabel("Coefficient Value")
        axs[i].set_ylabel("Frequency")

    # Hide any extra subplots.
    for j in range(len(mask_columns), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    hist_path = os.path.join(save_folder, "mask_coefficients_histograms.png")
    plt.savefig(hist_path)
    plt.close(fig)
    print(f"Histograms saved to {hist_path}")

def plot_correlation_heatmap(csv_filename, save_folder):
    """
    Plot a correlation heatmap of the mask coefficients.
    """
    if not os.path.exists(csv_filename):
        print(f"CSV file {csv_filename} not found!")
        return

    # Read the CSV file.
    df = pd.read_csv(csv_filename)
    mask_columns = [col for col in df.columns if col.startswith("mask_")]

    # Compute the correlation matrix.
    corr = df[mask_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    # Set tick marks and labels.
    ax.set_xticks(range(len(mask_columns)))
    ax.set_yticks(range(len(mask_columns)))
    ax.set_xticklabels(mask_columns, rotation=90, fontsize=8)
    ax.set_yticklabels(mask_columns, fontsize=8)
    ax.set_title("Correlation Heatmap of Mask Coefficients", pad=20)

    plt.tight_layout()
    heatmap_path = os.path.join(save_folder, "mask_coefficients_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")

def main():
    csv_filename = "mask_coefficients.csv"
    save_folder = "visualizations"
    
    # Ensure the save folder exists.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Generate and save the visualizations.
    plot_histograms(csv_filename, save_folder)
    plot_correlation_heatmap(csv_filename, save_folder)

if __name__ == "__main__":
    main()

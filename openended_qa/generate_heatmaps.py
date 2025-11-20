import numpy as np
import matplotlib.pyplot as plt
import os

# Make directory for saving heatmaps
os.makedirs("heatmaps", exist_ok=True)

# --------------------------------------------
# 1. Raw data (Make sure to replace the below to your own data)
# --------------------------------------------
data = [
    [0.0, 0.0, 0.00450, 1.000, 1.000, 0.630, 0.724, 0.016],
    [0.0, 0.5, 0.04863, 0.500, 0.500, 0.630, 0.722, 0.016],
    [0.0, 0.9, 0.04866, 0.100, 0.100, 0.630, 0.724, 0.016],

    [0.3, 0.0, 0.01662, 0.953, 0.967, 0.626, 0.732, 0.014],
    [0.3, 0.5, 0.01713, 0.453, 0.467, 0.632, 0.730, 0.018],
    [0.3, 0.9, 0.03320, 0.051, 0.0667, 0.628, 0.722, 0.010],

    [0.7, 0.0, 0.00819, 0.850, 0.883, 0.628, 0.694, 0.020],
    [0.7, 0.5, 0.00629, 0.351, 0.381, 0.620, 0.708, 0.020],
    [0.7, 0.9, 0.04959, 0.000, 0.00041, 0.614, 0.682, 0.016],

    [1.0, 0.0, 0.00509, 0.759, 0.798, 0.632, 0.606, 0.012],
    [1.0, 0.5, 0.01508, 0.248, 0.295, 0.618, 0.620, 0.006],
    [1.0, 0.9, 0.04996, 0.000, 0.00004, 0.608, 0.642, 0.006],
]

arr = np.array(data)

temperatures = sorted(list(set(arr[:,0])))
perturbations = sorted(list(set(arr[:,1])))

# --------------------------------------------
# 2. Convert raw table → heatmap matrices
# --------------------------------------------
def build_matrix(column_index):
    M = np.zeros((len(temperatures), len(perturbations)))
    for row in arr:
        T, P = row[0], row[1]
        i = temperatures.index(T)
        j = perturbations.index(P)
        M[i, j] = row[column_index]
    return M

ECE90  = build_matrix(2)
QRC90  = build_matrix(3)
CVAR90 = build_matrix(4)
ACC_S  = build_matrix(5)
ACC_T  = build_matrix(6)
ACC_TR = build_matrix(7)

# --------------------------------------------
# 3. Plot helper with save support
# --------------------------------------------
def plot_heatmap(values, title, filename):
    plt.figure(figsize=(6,5))
    plt.imshow(values, cmap="viridis", aspect='auto')
    plt.colorbar(label=title)
    plt.xticks(ticks=np.arange(len(perturbations)), labels=perturbations)
    plt.yticks(ticks=np.arange(len(temperatures)), labels=temperatures)
    plt.xlabel("Perturbation")
    plt.ylabel("Temperature")
    plt.title(title)
    plt.tight_layout()

    # Save to file
    save_path = os.path.join("heatmaps", filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

# --------------------------------------------
# 4. Generate and save all 6 heatmaps
# --------------------------------------------
plot_heatmap(ECE90,  "ECE90 Heatmap",  "ECE90.png")
plot_heatmap(QRC90,  "QRC90 Heatmap",  "QRC90.png")
plot_heatmap(CVAR90, "CVAR90 Heatmap", "CVAR90.png")

plot_heatmap(ACC_S,  "SQuAD Accuracy Heatmap",      "SQuAD_Accuracy.png")
plot_heatmap(ACC_T,  "TriviaQA Accuracy Heatmap",   "TriviaQA_Accuracy.png")
plot_heatmap(ACC_TR, "TruthfulQA Accuracy Heatmap", "TruthfulQA_Accuracy.png")

import matplotlib.pyplot as plt
import torch
import numpy as np

# ----------------- Function to Plot Contour -----------------
def plot_contour(true_solution, predicted_solution, sample_idx=0):
    """
    Plots the contour plots of the true and predicted solutions for a given test sample.

    :param true_solution: Ground truth u_test tensor
    :param predicted_solution: Model output tensor u_pred
    :param sample_idx: Index of the sample to visualize
    """

    # Convert tensors to NumPy for plotting
    true_solution_np = true_solution[sample_idx].cpu().numpy()
    predicted_solution_np = predicted_solution[sample_idx].cpu().numpy()

    # Get the grid dimensions
    grid_size_x, grid_size_y = true_solution_np.shape  # Assuming square or rectangular grid

    # Define extent so that x and y range from 0 to 1
    extent = [0, 1, 0, 1]

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot True Solution
    ax1 = axes[0]
    c1 = ax1.contourf(true_solution_np, levels=20, cmap="coolwarm", extent=extent)
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("True Solution", fontsize=15)
    ax1.set_xlabel("x-coordinate", fontsize=14)
    ax1.set_ylabel("y-coordinate", fontsize=14)

    # Plot Predicted Solution
    ax2 = axes[1]
    c2 = ax2.contourf(predicted_solution_np, levels=20, cmap="coolwarm", extent=extent)
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Approximate Solution (CNN)", fontsize=15)
    ax2.set_xlabel("x-coordinate", fontsize=14)
    ax2.set_ylabel("y-coordinate", fontsize=14)
    
    
    ax1.tick_params(axis='x', labelsize=12)  # X-axis tick label font size
    ax1.tick_params(axis='y', labelsize=12)  # Y-axis tick label font size
    ax2.tick_params(axis='x', labelsize=12)  # X-axis tick label font size
    ax2.tick_params(axis='y', labelsize=12)  # Y-axis tick label font size

    plt.tight_layout()
    plt.show()

# ----------------- Select a Sample from Test Data -----------------
sample_idx = 0  # Change this to visualize different test samples

# Ensure the model is in evaluation mode
net.eval()

# Get test sample
a_sample = a_test[sample_idx].unsqueeze(0)  # Add batch dimension

# Forward pass to get the predicted solution
with torch.no_grad():
    u_pred = net(a_sample)
    u_pred = u_normalizer.decode(u_pred)  # Decode to original scale

# ----------------- Plot the Contour Plots -----------------
plot_contour(u_test, u_pred, sample_idx)


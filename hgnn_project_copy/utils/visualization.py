# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, save_path=None):
    """Plot training and test loss curves"""
    epochs = range(1, len(train_losses) + 1)
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
   
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
   
    plt.tight_layout()
   
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
    plt.show()
    return fig

def visualize_graph(data, title="Graph Visualization", save_path=None):
    """Visualize a graph in 3D"""
    if torch.is_tensor(data.x):
        positions = data.x.detach().cpu().numpy()
    else:
        positions = data.x
   
    if torch.is_tensor(data.edge_index):
        edge_index = data.edge_index.detach().cpu().numpy()
    else:
        edge_index = data.edge_index
   
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
   
    # Plot nodes
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c='red', s=50, alpha=0.8)
   
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        ax.plot([positions[src, 0], positions[dst, 0]],
                [positions[src, 1], positions[dst, 1]],
                [positions[src, 2], positions[dst, 2]],
                'b-', alpha=0.3, linewidth=0.5)
   
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
    plt.show()
    return fig

def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """Visualize a 3D point cloud"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
   
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
   
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
              c='blue', s=20, alpha=0.6)
   
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   
    # Make axes equal
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
   
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
   
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
   
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
    plt.show()
    return fig

def plot_class_distribution(dataset, class_names=None, save_path=None):
    """Plot class distribution in dataset"""
    labels = []
    for i in range(len(dataset)):
        data = dataset[i]
        labels.append(data.y.item() if torch.is_tensor(data.y) else data.y)
   
    unique_labels, counts = np.unique(labels, return_counts=True)
   
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
   
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique_labels)), counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(len(unique_labels)), class_names, rotation=45)
   
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
   
    plt.tight_layout()
   
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
    plt.show()
    return plt.gcf()

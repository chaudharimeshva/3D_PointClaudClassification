
import os
import urllib.request
import zipfile
import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
import pickle
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler

def download_modelnet10(dest_folder="data"):
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = os.path.join(dest_folder, "ModelNet10.zip")
    modelnet_path = os.path.join(dest_folder, "ModelNet10")

    os.makedirs(dest_folder, exist_ok=True)

    if not os.path.exists(modelnet_path):
        print("Downloading ModelNet10...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)

        os.remove(zip_path)
    else:
        print("ModelNet10 already exists.")
    return modelnet_path

def load_off_mesh(file_path):
    """Load OFF mesh file with error handling"""
    try:
        mesh = trimesh.load(file_path, process=False)
        if mesh.is_empty:
            return None
        return mesh
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None

def mesh_to_point_cloud(mesh, num_points=512):
    """Convert mesh to point cloud with better sampling"""
    try:
        # Use surface sampling for better point distribution
        points = mesh.sample(num_points)
    
        
        # Add some noise to prevent identical points
        noise = np.random.normal(0, 0.001, points.shape)
        points = points + noise
        
        return points
    except Exception as e:
        print(f"Error sampling points: {e}")
        return None

def build_knn_graph(points, k=10):
    """Build k-NN graph with improved connectivity"""
    kdtree = KDTree(points)
    edges = []
    weights = []
    
    for i, point in enumerate(points):
        # Find k+1 nearest neighbors (including self)
        distances, indices = kdtree.query(point, k=k+1)
        
        # Skip the first one (self)
        for j, neighbor_idx in enumerate(indices[1:]):
            edges.append((i, neighbor_idx))
            # Use inverse distance as weight (add small epsilon to avoid division by zero)
            weight = 1.0 / (distances[j+1] + 1e-6)
            weights.append(weight)
    
    return edges, weights

def add_geometric_features(points):
    """Add geometric features to point cloud"""
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Add distance from centroid as feature
    distances = np.linalg.norm(centered_points, axis=1, keepdims=True)
    
    # Add local density feature
    kdtree = KDTree(points)
    densities = []
    for point in points:
        # Count neighbors within a small radius
        neighbors = kdtree.query_ball_point(point, r=0.1)
        densities.append(len(neighbors))
    
    densities = np.array(densities).reshape(-1, 1)
    
    # Normalize features
    scaler = StandardScaler()
    distances = scaler.fit_transform(distances)
    densities = scaler.fit_transform(densities)
    
    # Combine original coordinates with new features
    enhanced_features = np.concatenate([centered_points, distances, densities], axis=1)
    enhanced_features = (enhanced_features - enhanced_features.mean(axis=0)) / (enhanced_features.std(axis=0) + 1e-8)
    enhanced_features = np.nan_to_num(enhanced_features)

    
    return enhanced_features

def construct_graph(points, k=10, add_features=True):
    """Construct graph with enhanced features"""
    edges, weights = build_knn_graph(points, k)
    
    G = nx.Graph()
    
    # Add geometric features if requested
    if add_features:
        features = add_geometric_features(points)
    else:
        # Just center the points
        features = points - np.mean(points, axis=0)
    
    # Add nodes with features
    for i, feat in enumerate(features):
        G.add_node(i, x=feat)
    
    # Add weighted edges
    for (i, j), weight in zip(edges, weights):
        G.add_edge(i, j, weight=weight)
    
    return G

def graph_to_matrices(G):
    """Convert graph to matrices with proper handling"""
    num_nodes = G.number_of_nodes()
    
    # Extract node features
    features = np.array([G.nodes[i]['x'] for i in range(num_nodes)])
    
    # Build adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    for i, j, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        adj[i, j] = weight
        adj[j, i] = weight
    
    return adj, features

def save_graph_pickle(adj, features, label, out_path):
    """Save graph data to pickle file"""
    obj = {
        "adj": adj.astype(np.float32),  # Use float32 to save memory
        "features": features.astype(np.float32),
        "label": int(label)
    }
    with open(out_path, 'wb') as f:
        pickle.dump(obj, f)

def process_modelnet10_to_graphs_pickle(root_dir, out_dir="data/graph_data", k=10, num_points=512, add_features=True):
    """Process ModelNet10 dataset to graph format"""
    all_class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_map = {name: i for i, name in enumerate(all_class_names)}

    print(f"Found classes: {list(class_map.keys())}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Save class mapping
    with open(os.path.join(out_dir, "label_map.json"), 'w') as f:
        json.dump(class_map, f, indent=2)

    stats = {"train": 0, "test": 0, "failed": 0}
    
    for split in ['train', 'test']:
        print(f"\nProcessing {split} data...")
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_name in tqdm(class_map.keys(), desc=f"{split}"):
            class_path = os.path.join(root_dir, class_name, split)
            if not os.path.exists(class_path):
                continue
                
            files = [f for f in os.listdir(class_path) if f.endswith(".off")]

            for file in files:
                file_path = os.path.join(class_path, file)
                try:
                    # Load mesh
                    mesh = load_off_mesh(file_path)
                    if mesh is None:
                        stats["failed"] += 1
                        continue
                    
                    # Convert to point cloud
                    points = mesh_to_point_cloud(mesh, num_points)
                    if points is None:
                        stats["failed"] += 1
                        continue
                    
                    # Construct graph
                    G = construct_graph(points, k, add_features)
                    
                    # Convert to matrices
                    adj, features = graph_to_matrices(G)
                    
                    # Save
                    out_file = os.path.join(split_dir, f"{class_name}_{file[:-4]}.pkl")
                    save_graph_pickle(adj, features, class_map[class_name], out_file)
                    
                    stats[split] += 1
                    
                except Exception as e:
                    print(f"Failed on {file_path}: {e}")
                    stats["failed"] += 1

    print(f"\nProcessing complete!")
    print(f"Train samples: {stats['train']}")
    print(f"Test samples: {stats['test']}")
    print(f"Failed samples: {stats['failed']}")

if __name__ == "__main__":
    modelnet_path = download_modelnet10()
    process_modelnet10_to_graphs_pickle(
        modelnet_path, 
        out_dir="data/graph_data", 
        k=15,  # Increased k for better connectivity
        num_points=512,
        add_features=True  # Add geometric features
    )
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from models.hgnn import HGNN
from data.modelnet_dataset import ModelNetGraphDataset
from utils.metrics import compute_metrics
import yaml
import argparse
import os
import json
from tqdm import tqdm

def test_model(model, loader, device, class_names=None):
    """Test the model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            out = global_mean_pool(out, data.batch)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)
    return metrics, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/modelnet10_graphs.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = ModelNetGraphDataset(root=os.path.join(cfg['data_root'], 'test'))
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)}")

    # Load class names
    label_map_path = os.path.join(cfg['data_root'], 'label_map.json')
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    else:
        class_names = [f"Class_{i}" for i in range(test_dataset.num_classes)]

    # Initialize model
    model = HGNN(
        input_dim=test_dataset.num_features,
        hidden_dim=cfg['hidden_dim'],
        num_classes=test_dataset.num_classes,
        num_layers=cfg.get('num_layers', 2),
        c=cfg.get('curvature', 1.0)
    ).to(device)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"Checkpoint {args.checkpoint} not found!")
        return

    # Test the model
    metrics, preds, labels = test_model(model, test_loader, device, class_names)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    print("\nPer-class Results:")
    print("-" * 30)
    for i, class_name in enumerate(class_names):
        precision = metrics['per_class_precision'][i]
        recall = metrics['per_class_recall'][i]
        f1 = metrics['per_class_f1'][i]
        print(f"{class_name:15}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Save results if requested
    if args.save_results:
        os.makedirs('results', exist_ok=True)
        
        # Save metrics
        with open('results/test_metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {}
            for k, v in metrics.items():
                if hasattr(v, 'tolist'):
                    metrics_json[k] = v.tolist()
                else:
                    metrics_json[k] = v
            json.dump(metrics_json, f, indent=2)
        
        # Save predictions
        results_data = {
            'predictions': preds,
            'ground_truth': labels,
            'class_names': class_names
        }
        with open('results/predictions.json', 'w') as f:
            # Convert numpy types to native Python types
            results_data = {
                'predictions': [int(p) for p in preds],
                'ground_truth': [int(l) for l in labels],
                'class_names': class_names
            }
            json.dump(results_data, f, indent=2)

        
        print(f"\nResults saved to 'results/' directory")

if __name__ == '__main__':
    main()
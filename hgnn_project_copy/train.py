import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from models.hgnn import HGNN
from data.modelnet_dataset import ModelNetGraphDataset
from utils.visualization import plot_training_curves
import torch.nn.functional as F
import yaml
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
   
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
       
        # Forward pass
        out = model(data.x, data.edge_index, data.batch)
       
        # Global pooling for graph-level prediction
        out = global_mean_pool(out, data.batch)
       
        # Calculate loss
        loss = F.cross_entropy(out, data.y)
        loss.backward()
       
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       
        optimizer.step()
       
        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total_samples += data.y.size(0)
        total_loss += loss.item()
   
    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples
   
    return avg_loss, accuracy

def test(model, loader, device):
    model.eval()
    correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            out = global_mean_pool(out, data.batch)
            loss = F.cross_entropy(out, data.y)
            total_loss += loss.item() * data.y.size(0)

            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_samples += data.y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='hgnn_project_copy/config/modelnet10_graphs.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load datasets
    train_dataset = ModelNetGraphDataset(root=os.path.join(cfg['data_root'], 'train'))
    test_dataset = ModelNetGraphDataset(root=os.path.join(cfg['data_root'], 'test'))
   
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of features: {train_dataset.num_features}")
    print(f"Number of classes: {train_dataset.num_classes}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    # Initialize model
    model = HGNN(
        input_dim=train_dataset.num_features,
        hidden_dim=cfg['hidden_dim'],
        num_classes=train_dataset.num_classes,
        num_layers=cfg.get('num_layers', 2),
        c=cfg.get('curvature', 1.0)
    ).to(device)
   
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(cfg['lr']),
    weight_decay=float(cfg.get('weight_decay', 5e-4))
)
   
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get('step_size', 50), gamma=cfg.get('gamma', 0.5))

    # Training history
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
   
    best_acc = 0.0
    start_epoch = 0
   
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        test_accs = checkpoint.get('test_accs', [])

    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
       
        # Training
        train_loss, train_acc = train(model, train_loader, optimizer, device)
       
        # Testing
        test_loss, test_acc = test(model, test_loader, device)
       
        # Update learning rate
        scheduler.step()
       
        # Store history
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
       
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
       
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }
       
        torch.save(checkpoint, 'checkpoints/last_checkpoint.pth')
       
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            torch.save(checkpoint, 'checkpoints/best_checkpoint.pth')
            print(f"New best accuracy: {best_acc:.4f}")
       
        # Save training log
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'lr': scheduler.get_last_lr()[0]
        }
       
        # Append to log file
        with open('logs/training_log.jsonl', 'a') as f:
            f.write(json.dumps(log_data) + '\n')

    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.4f}")
   
    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, 'results/training_curves.png')
   
    # Save final training history
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc,
        'config': cfg
    }
   
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main()
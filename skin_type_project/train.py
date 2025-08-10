import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from datasets.skintype import SkinType
from models.efficientnet_model import get_model
from utils.transforms import get_transforms
from torch.utils.tensorboard import SummaryWriter

def plot_training(train_losses, val_losses, train_accs, val_accs, save_path='training_curve.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='red')
    plt.plot(epochs, val_losses, label='Val Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved training curve to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved at: {save_path}")

def evaluate(model, loader, criterion, device, return_details=False):
    model.eval()
    total, correct, loss_total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss_total += criterion(logits, labels).item() * labels.size(0)
            _, pred = logits.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    loss = loss_total / total
    if return_details:
        return acc, loss, all_preds, all_labels
    return acc, loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    train_ds = SkinType("crop_yolo_new2", split="train", transform=get_transforms(True))#83%
    val_ds = SkinType("crop_yolo_new2", split="valid", transform=get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=80)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    writer = SummaryWriter(log_dir="runs/efficientnet_b0_skin")

    best_acc = 0
    patience = 5
    counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(80):
        model.train()
        total, correct, loss_total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, pred = logits.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item() * labels.size(0)

        train_acc = correct / total
        train_loss = loss_total / total
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_model_path)
            print(f"✅ Best model saved at: {args.save_model_path}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("🚨 Early stopping triggered.")
                break

    plot_training(train_losses, val_losses, train_accs, val_accs, save_path=args.output_plot)
    print(f"🌟 Highest Validation Accuracy: {best_acc * 100:.2f}%")

    # Final evaluation
    val_acc, val_loss, all_preds, all_labels = evaluate(model, val_loader, criterion, device, return_details=True)
    class_names = ["Oily", "Dry", "Normal"]  # sửa nếu bạn dùng lớp khác

    print("\n📄 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(all_labels, all_preds, class_names, save_path='confusion_matrix.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-plot', type=str, default='training_curve.png')
    parser.add_argument('--save-model-path', type=str, default='best_model.pth')
    args = parser.parse_args()
    train(args)

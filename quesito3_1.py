

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
TRAIN_DIR   = "CVPR2023_project_2_and_3_data/train"
TEST_DIR    = "CVPR2023_project_2_and_3_data/test"   
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 0.001
NUM_CLASSES = 15
SEED        = 42
IMG_SIZE    = 224


torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────
# 3. DATASET — Train 85% / Val 15% + Test
# ─────────────────────────────────────────
dataset_for_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
dataset_for_val   = datasets.ImageFolder(TRAIN_DIR, transform=val_test_transform)
CLASS_NAMES = dataset_for_train.classes
targets = dataset_for_train.targets

# Split 85% Train / 15% Validation (stratificato)
n_total = len(dataset_for_train)
n_val   = int(n_total * 0.15)
n_train = n_total - n_val
generator = torch.Generator().manual_seed(SEED)
train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
train_indices = list(train_idx)
val_indices   = list(val_idx)

# Creiamo i Subset
train_dataset = Subset(dataset_for_train, train_indices)
val_dataset   = Subset(dataset_for_val, val_indices)

# Carichiamo il vero Test Set dalla nuova cartella
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
print("Classi nel TEST  set:", test_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

# ─────────────────────────────────────────
# 4. MODELLO — AlexNet con freeze
# ─────────────────────────────────────────
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# Congela TUTTI i parametri
for param in model.parameters():
    param.requires_grad = False

# Sostituisci SOLO l'ultimo FC layer
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
model = model.to(device)

print("\nParametri trainable:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"  ✓ {name}")

# ─────────────────────────────────────────
# 5. LOSS & OPTIMIZER
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, momentum=0.9,weight_decay=1e-3
)

# ─────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────
def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print("\nInizio addestramento...")
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        vl_loss, vl_acc = run_epoch(val_loader,   train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        # Salviamo il modello migliore basato sul Validation Set
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "CVPR2023_project_2_and_3_data/alexnet_freeze_best.pth")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    # ─────────────────────────────────────────
    # 7. PLOTS (TRAIN vs VAL)
    # ─────────────────────────────────────────
    epochs_range = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val")
    axes[0].set_title("Loss (Train vs Val) — AlexNet Freeze"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_acc"], label="Train")
    axes[1].plot(epochs_range, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy (Train vs Val) — AlexNet Freeze"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("CVPR2023_project_2_and_3_data/alexnet_freeze_curves.png", dpi=150)
    plt.show()

    # ─────────────────────────────────────────
    # 8. TEST FINALE SUL NUOVO DATASET
    # ─────────────────────────────────────────
    print("\n--- Valutazione finale sul TEST SET ---")
    model.load_state_dict(torch.load("CVPR2023_project_2_and_3_data/alexnet_freeze_best.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:   
            images = images.to(device)
            preds  = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy (su Test Set): {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # Confusion Matrix sul Test Set
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:\n{cm}")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix sul TEST SET  (Acc: {test_acc*100:.1f}%)')
    plt.tight_layout()
    plt.savefig("CVPR2023_project_2_and_3_data/alexnet_freeze_test_confusion.png", dpi=150)
    plt.show()
    print("Saved: alexnet_freeze_test_confusion.png")

    

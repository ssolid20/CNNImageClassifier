

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
DATA_DIR_TRAIN = "CVPR2023_project_2_and_3_data/train"   # cartella train con 15 sottocartelle
DATA_DIR_TEST  = "CVPR2023_project_2_and_3_data/test"    # cartella test  con 15 sottocartelle
IMG_SIZE       = (64, 64)
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LR             = 0.001
VAL_SPLIT      = 0.15       # 85% train, 15% validation
NUM_CLASSES    = 15
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────
# Augmentation solo per il training (flip orizzontale)
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
   # transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),     # riporta in [0,255] !!
    #transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Val e test: solo resize, nessuna augmentation
eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),     # riporta in [0,255] !!
    #transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ─────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────

# --- 3A. TEST SET (già separato su disco, NON si tocca) ---
test_dataset = datasets.ImageFolder(DATA_DIR_TEST, transform=eval_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test set:  {len(test_dataset)} immagini")

# --- 3B. TRAINING FULL caricato due volte con transform diverse ---
# (stesso trucco di prima: stessi indici, transform diverse)
train_full_aug  = datasets.ImageFolder(DATA_DIR_TRAIN, transform=train_transform)
train_full_eval = datasets.ImageFolder(DATA_DIR_TRAIN, transform=eval_transform)

n_total = len(train_full_aug)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

# Split casuale riproducibile con seed
generator = torch.Generator().manual_seed(SEED)
train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)
train_indices = list(train_indices)
val_indices   = list(val_indices)

# Subset con le rispettive transform
train_dataset = Subset(train_full_aug,  train_indices)   # con flip
val_dataset   = Subset(train_full_eval, val_indices)     # senza flip

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train:     {len(train_dataset)} immagini  ({100*(1-VAL_SPLIT):.0f}%)")
print(f"Val:       {len(val_dataset)}  immagini  ({100*VAL_SPLIT:.0f}%)")
print(f"Classi:    {train_full_aug.classes}")

# ─────────────────────────────────────────
# 4. MODEL (Table 1 – layout esatto)
# ─────────────────────────────────────────
def inizializza_pesi(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class ShallowCNN(nn.Module):
    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8,  kernel_size=3, stride=1, padding=1),  # layer 2
            nn.ReLU(inplace=True),                                  # layer 3
            nn.MaxPool2d(kernel_size=2, stride=2),                  # layer 4
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # layer 5
            nn.ReLU(inplace=True),                                  # layer 6
            nn.MaxPool2d(kernel_size=2, stride=2),                  # layer 7
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # layer 8
            nn.ReLU(inplace=True),                                  # layer 9
        )
        # shape dopo block3: [B, 32, 16, 16] → flatten → 8192
        self.fc      = nn.Linear(32 * 16 * 16, num_classes)        # layer 10
        self.softmax = nn.Softmax(dim=1)                            # layer 11 (solo inference)

    def forward(self, x, return_logits: bool = True):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)      # flatten
        x = self.fc(x)                  # logits
        if not return_logits:
            x = self.softmax(x)
        return x

model = ShallowCNN(num_classes=NUM_CLASSES).to(device)
model.apply(inizializza_pesi)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ─────────────────────────────────────────
# 5. LOSS & OPTIMIZER
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# ─────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────
def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, return_logits=True)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

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

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "CVPR2023_project_2_and_3_data/best_model_quesito1.pth")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # ─────────────────────────────────────────
    # 7. GRAFICI LOSS & ACCURACY (train + val)
    # ─────────────────────────────────────────
    epochs_range = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val")
    axes[0].set_title("Loss");     axes[0].set_xlabel("Epoch")
    axes[0].legend();              axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_acc"], label="Train")
    axes[1].plot(epochs_range, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend();              axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("CVPR2023_project_2_and_3_data/training_curvesQuesito1.png", dpi=150)
    plt.show()
    print("Grafici salvati in training_curves.png")

    # ─────────────────────────────────────────
    # 8. VALUTAZIONE FINALE SUL TEST SET
    # ─────────────────────────────────────────
    # Carica il miglior modello salvato durante il training
    model.load_state_dict(torch.load("CVPR2023_project_2_and_3_data/best_model_quesito1.pth", map_location=device))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:          # <-- test_loader, NON val_loader
            images  = images.to(device)
            outputs = model(images, return_logits=True)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Overall accuracy sul test set
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Set — Overall Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # Confusion matrix sul test set
    cm          = confusion_matrix(all_labels, all_preds)
    class_names = test_dataset.classes   # nomi dalla cartella test

    print("\nConfusion Matrix (test set):")
    print(cm)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — Test Set  (accuracy: {test_acc:.4f})')
    plt.tight_layout()
    plt.savefig("CVPR2023_project_2_and_3_data/confusion_matrix_testsetQuesito1.png", dpi=150)
    plt.show()
    print("Confusion matrix salvata in confusion_matrix_testset.png")


   
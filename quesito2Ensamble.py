
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR      = "CVPR2023_project_2_and_3_data/train"
DATA_DIR_TEST = "CVPR2023_project_2_and_3_data/test"
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 0.001
NUM_CLASSES = 15
SEED        = 42
N_MODELS    = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),     
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),     
])

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
train_full_aug  = datasets.ImageFolder(DATA_DIR, transform=train_transform)
train_full_eval = datasets.ImageFolder(DATA_DIR, transform=val_transform)

n_total = len(train_full_aug)
n_val   = int(n_total * 0.15)
n_train = n_total - n_val
generator = torch.Generator().manual_seed(SEED)
train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
train_indices = list(train_idx)
val_indices   = list(val_idx)

train_dataset = Subset(train_full_aug,  train_indices)
val_dataset   = Subset(train_full_eval, val_indices)

test_dataset  = datasets.ImageFolder(DATA_DIR_TEST, transform=val_transform)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader    = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class ShallowCNN(nn.Module):
    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.58)
        self.fc      = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # logits



def inizializza_pesi(m):
    # Applica la regola solo ai livelli Conv2d e Linear
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # 1. Pesi: Distribuzione Gaussiana (Normale) con media 0 e dev. std 0.01
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        
        # 2. Bias: Impostati esattamente a 0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
# ─────────────────────────────────────────
# FUNZIONE: train singola rete
# ─────────────────────────────────────────
def train_single_model(model_id: int):
    print(f"\n{'='*50}")
    print(f"  Training model {model_id + 1}/{N_MODELS}")
    print(f"{'='*50}")

    torch.manual_seed(SEED + model_id * 100)
    np.random.seed(SEED + model_id * 100)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model     = ShallowCNN(num_classes=NUM_CLASSES).to(device)
    model.apply(inizializza_pesi)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        # VAL
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits  = model(images)
                loss    = criterion(logits, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        tr_acc = train_correct / train_total
        vl_acc = val_correct   / val_total

        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss/train_total:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss/val_total:.4f}  val_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"  → Best val_acc model {model_id+1}: {best_val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, best_val_acc


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    ensemble_models = []
    individual_accs = []
    class_names     = test_dataset.classes
    softmax         = nn.Softmax(dim=1)

    for i in range(N_MODELS):
        trained_model, acc = train_single_model(i)
        trained_model.eval()
        ensemble_models.append(trained_model)
        individual_accs.append(acc)
        torch.save(trained_model.state_dict(), f"CVPR2023_project_2_and_3_data/model_{i+1}.pth")

    print(f"\nAccuratezze individuali (test): {[f'{a:.4f}' for a in individual_accs]}")
    print(f"Media singole reti (val):        {np.mean(individual_accs):.4f}")

    # ─────────────────────────────────────────
    # PREDIZIONE ENSEMBLE — SUL TEST SET
    # media aritmetica delle probabilità softmax 
    # ─────────────────────────────────────────
    all_labels         = []
    all_ensemble_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images    = images.to(device)
            avg_probs = torch.zeros(images.size(0), NUM_CLASSES).to(device)
            for model in ensemble_models:
                avg_probs += softmax(model(images))
            avg_probs /= N_MODELS
            preds = avg_probs.argmax(dim=1)
            all_ensemble_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    ensemble_acc = accuracy_score(all_labels, all_ensemble_preds)
    print(f"\nEnsemble accuracy ({N_MODELS} modelli) — TEST SET: {ensemble_acc:.4f}  "
          f"({ensemble_acc*100:.1f}%)")
    print(f"Miglioramento vs media singole (val): "
          f"+{(ensemble_acc - np.mean(individual_accs))*100:.1f}%")

    # ─────────────────────────────────────────
    # CONFUSION MATRIX ENSEMBLE — TEST SET
    # ─────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_ensemble_preds)
    print(f"\nEnsemble Confusion Matrix:\n{cm}")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Ensemble Confusion Matrix ({N_MODELS} models) — '
                 f'Test Set Accuracy: {ensemble_acc*100:.1f}%')
    plt.tight_layout()
    plt.savefig(f"CVPR2023_project_2_and_3_data/ensemble_confusion_matrix_testset{N_MODELS}.png", dpi=150)
    plt.show()
    print(f"Saved: ensemble_confusion_matrix_testset{N_MODELS}.png")

    # ─────────────────────────────────────────
    # CONFRONTO: singole reti vs ensemble
    # ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [f"Model {i+1}" for i in range(N_MODELS)] + ["Ensemble"],
        individual_accs + [ensemble_acc],
        color=["steelblue"] * N_MODELS + ["tomato"]
    )
    ax.axhline(np.mean(individual_accs), color="gray", linestyle="--", label="Media singole reti")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Singole reti vs Ensemble — Test Set")
    ax.legend()
    for bar, val in zip(bars, individual_accs + [ensemble_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"CVPR2023_project_2_and_3_data/ensemble_comparison{N_MODELS}.png", dpi=150)
    plt.show()
    print("Saved: ensemble_comparison.png")


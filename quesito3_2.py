
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
DATA_DIR      = "CVPR2023_project_2_and_3_data/train"
DATA_DIR_TEST = "CVPR2023_project_2_and_3_data/test"   
BATCH_SIZE  = 32
NUM_CLASSES = 15
SEED        = 42
IMG_SIZE    = 224   # AlexNet richiede 224×224

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# 2. TRANSFORMS (nessun flip su val/test)
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────
# 3. DATASET — 85% train / 15% val + test set separato
# ─────────────────────────────────────────
train_full_aug  = datasets.ImageFolder(DATA_DIR, transform=train_transform)
train_full_eval = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

n_total = len(train_full_aug)
n_val   = int(n_total * 0.15)
n_train = n_total - n_val
generator = torch.Generator().manual_seed(SEED)
train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
train_indices = list(train_idx)
val_indices   = list(val_idx)

train_dataset = Subset(train_full_aug,  train_indices)   # con flip
val_dataset   = Subset(train_full_eval, val_indices)     # senza flip

# Test set separato — mai visto durante il training
test_dataset  = datasets.ImageFolder(DATA_DIR_TEST, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ─────────────────────────────────────────
# 4. MODELLO — AlexNet come feature extractor
# Sostituisce l'ultimo layer con Identity → output 4096-dim
# ─────────────────────────────────────────
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Identity()   # output: vettore 4096-dim
model = model.to(device)
model.eval()

print(f"Parametri trainable: 0 (0.0%) — solo feature extraction")

# ─────────────────────────────────────────
# 5. ESTRAZIONE FEATURES
# ─────────────────────────────────────────
def extract_features(loader, desc=""):
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images   = images.to(device)
            features = model(images)                    # [B, 4096]
            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    X = np.array(all_features)
    y = np.array(all_labels)
    if desc:
        print(f"  {desc}: {X.shape}")
    return X, y


if __name__ == "__main__":
    print("\nEstrazione features...")
    X_train, y_train = extract_features(train_loader, "Train")
    X_val,   y_val   = extract_features(val_loader,   "Val  ")
    X_test,  y_test  = extract_features(test_loader,  "Test ")

    # ─────────────────────────────────────────
    # 6. SVM LINEARE (One-vs-One, come da PDF)
    # ─────────────────────────────────────────
    print("\nAddestramento SVM lineare (OVO)...")
    svm = SVC(kernel='linear', decision_function_shape='ovo', random_state=SEED)
    svm.fit(X_train, y_train)
    print("SVM addestrata.")

    # ─────────────────────────────────────────
    # 7. ACCURACY su train, val e test
    # ─────────────────────────────────────────
    train_acc = accuracy_score(y_train, svm.predict(X_train))
    val_acc   = accuracy_score(y_val,   svm.predict(X_val))

    y_pred_test = svm.predict(X_test)
    test_acc    = accuracy_score(y_test, y_pred_test)

    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"Val   Accuracy: {val_acc:.4f}   ({val_acc*100:.1f}%)")
    print(f"Test  Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # ─────────────────────────────────────────
    # 8. GRAFICI — accuracy su train / val / test
    # (la SVM non ha epoche, quindi usiamo un bar chart)
    # ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sets  = ["Train", "Val", "Test"]
    accs  = [train_acc, val_acc, test_acc]
    colors = ["steelblue", "orange", "tomato"]
    bars = ax.bar(sets, accs, color=colors)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("AlexNet Features + SVM — Accuracy per split")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CVPR2023_project_2_and_3_data/alexnet_svm_accuracy_splits.png", dpi=150)
    plt.show()
    print("Saved: alexnet_svm_accuracy_splits.png")

    # ─────────────────────────────────────────
    # 9. CONFUSION MATRIX + OVERALL ACCURACY — TEST SET
    # ─────────────────────────────────────────
    class_names = test_dataset.classes
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nConfusion Matrix (test set):\n{cm}")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'AlexNet Features + SVM — Confusion Matrix'
                 f'  (Test Acc: {test_acc*100:.1f}%)')
    plt.tight_layout()
    plt.savefig("CVPR2023_project_2_and_3_data/alexnet_svm_confusion_testset.png", dpi=150)
    plt.show()
    print("Saved: alexnet_svm_confusion_testset.png")


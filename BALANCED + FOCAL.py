import os
import json
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
 
# =========================
# CONFIG
# =========================
DATASET_PATH = r"C:\Users\SwathiSuresh\Downloads\Stage2_TypeRecognition.folder (1)"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 6
IMG = 240
SEED = 42
SAVE_AS = "best_c4.pt"
 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
 
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 
# =========================
# STEP 1: AUTO SPLIT (70/20/10) - same as senior
# =========================
def auto_split():
    print("\nChecking dataset...")
    train_path = os.path.join(DATASET_PATH, "train")
    val_path = os.path.join(DATASET_PATH, "val")
    test_path = os.path.join(DATASET_PATH, "test")
 
    if os.path.exists(val_path) and os.path.exists(test_path):
        print("Already split (train/val/test found), skipping split")
        return
 
    print("Creating train/val/test split (70/20/10)...")
    for cls in sorted(os.listdir(train_path)):
        cls_path = os.path.join(train_path, cls)
        files = sorted(os.listdir(cls_path))
        random.shuffle(files)
        total = len(files)
        train_end, val_end = int(0.7 * total), int(0.9 * total)
        os.makedirs(os.path.join(val_path, cls), exist_ok=True)
        os.makedirs(os.path.join(test_path, cls), exist_ok=True)
        for f in files[train_end:val_end]:
            shutil.move(os.path.join(cls_path, f), os.path.join(val_path, cls, f))
        for f in files[val_end:]:
            shutil.move(os.path.join(cls_path, f), os.path.join(test_path, cls, f))
    print("Split done (70/20/10)")
 
# =========================
# STEP 2: TRANSFORMS + LOADERS
# =========================
tf_train = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    NORM,
])
tf_eval = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    NORM,
])
 
def make_loader(ds, shuffle=False, sampler=None):
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=(shuffle if sampler is None else False),
                      sampler=sampler, num_workers=4, pin_memory=True)
 
# =========================
# COMBINATION: balanced sampling + focal loss
# =========================
GAMMA = 2.0
 
class FocalLoss(nn.Module):
    """Easy images count less, hard images count more. gamma=0 equals normal CE."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight
 
    def forward(self, logits, targets):
        ce_plain = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_plain)                       # probability of the true class
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1 - pt) ** self.gamma * ce).mean()
 
def build_train(train_path):
    ds = datasets.ImageFolder(train_path, transform=tf_train)
    labels = [l for _, l in ds.samples]
    counts = np.bincount(labels)
    print('Train images per class:', dict(zip(ds.classes, counts.tolist())))
    sample_w = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
    return ds, make_loader(ds, sampler=sampler)
 
def build_criterion(train_dataset):
    return FocalLoss(gamma=GAMMA, weight=None)
 
 
# =========================
# STEP 3: EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(DEVICE))
            preds += out.argmax(1).cpu().tolist()
            gts += labels.tolist()
    acc = float(np.mean(np.array(preds) == np.array(gts)))
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return acc, macro_f1, gts, preds
 
# =========================
# STEP 4: TRAIN
# =========================
def train_model():
    train_path = os.path.join(DATASET_PATH, "train")
    train_dataset, train_loader = build_train(train_path)
    classes = train_dataset.classes
    val_loader = make_loader(datasets.ImageFolder(os.path.join(DATASET_PATH, "val"), tf_eval))
    test_loader = make_loader(datasets.ImageFolder(os.path.join(DATASET_PATH, "test"), tf_eval))
    json.dump(classes, open("classes.json", "w"))
 
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(DEVICE)
 
    criterion = build_criterion(train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # same as senior
 
    print("Classes:", classes)
    print("\nTraining started (Combo 4: Balanced Sampling + Focal Loss)...")
    best_f1, no_improve = -1.0, 0
 
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
 
        acc, macro_f1, _, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch+1} | Mean loss: {total_loss/len(train_loader):.4f} "
              f"| Val acc: {acc:.4f} | Val macro-F1: {macro_f1:.4f}")
 
        if macro_f1 > best_f1:
            best_f1, no_improve = macro_f1, 0
            torch.save({"state": model.state_dict(), "classes": classes}, SAVE_AS)
            print("  Saved best model")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping (no improvement for {PATIENCE} epochs)")
                break
 
    # ---- final exam on the untouched test set ----
    ckpt = torch.load(SAVE_AS, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    acc, macro_f1, gts, preds = evaluate(model, test_loader)
    print("\n================ TEST RESULTS (Combo 4: Balanced Sampling + Focal Loss) ================")
    print(f"Test accuracy: {acc:.4f} | Test macro-F1: {macro_f1:.4f}")
    print(classification_report(gts, preds, target_names=classes, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(gts, preds))
 
if __name__ == "__main__":
    auto_split()
    train_model()
 

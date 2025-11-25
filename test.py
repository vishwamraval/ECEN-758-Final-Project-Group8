import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import DTD
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

def get_activation_fn(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(name)

def build_resnet18(num_classes, dropout, activation_name):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        get_activation_fn(activation_name),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

def get_predictions(model, loader):
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            all_labels.extend(y.numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def main():

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_set = DTD("data", split="test", download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    num_classes = len(test_set.classes)
    class_names = test_set.classes

    dropout = 0.0
    activation = "relu"

    model = build_resnet18(num_classes, dropout, activation).to(device)

    state_dict = torch.load("resnet18_dtd_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded model")

    test_labels, test_preds = get_predictions(model, test_loader)

    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    rec = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(test_labels, test_preds)

    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 15}
    )
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.title("Confusion Matrix for ResNet-18", fontsize=20)
    plt.xlabel("Predicted Class", fontsize=18)
    plt.ylabel("True Class", fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
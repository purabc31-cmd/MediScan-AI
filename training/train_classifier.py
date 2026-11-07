import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from models.classifier import build_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/classification")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    size = args.image_size
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train and validation class folders must match exactly.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(len(train_ds.classes))
    if args.pretrained:
        pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.load_state_dict({k: v for k, v in pretrained.state_dict().items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}, strict=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    Path("weights").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss/max(len(train_ds),1):.4f} | val_loss={val_loss/max(len(val_ds),1):.4f} | val_accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "weights/mri_classifier.pth")
            with open("weights/mri_classes.json", "w", encoding="utf-8") as handle:
                json.dump(train_ds.classes, handle, indent=2)

    print("Best validation accuracy:", best_accuracy)
    print("This is validation performance only. Evaluate once on an untouched patient-level test set before reporting results.")


if __name__ == "__main__":
    main()

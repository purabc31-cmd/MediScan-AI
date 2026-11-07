import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.unet import UNet2D


class PairedMaskDataset(Dataset):
    def __init__(self, root, split, size):
        self.images = Path(root) / "images" / split
        self.masks = Path(root) / "masks" / split
        self.paths = sorted(self.images.glob("*.png")) + sorted(self.images.glob("*.jpg")) + sorted(self.images.glob("*.jpeg"))
        self.size = size
        if not self.paths:
            raise ValueError(f"No images found in {self.images}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        mask_path = self.masks / (path.stem + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {path.name}: {mask_path}")
        image = Image.open(path).convert("L").resize((self.size, self.size))
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        if np.random.rand() < 0.5:
            x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])


def dice_score(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (denom + 1e-6)).mean().item()


class DiceBCELoss(nn.Module):
    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        dice = (2 * intersection + 1e-6) / (probs.sum() + target.sum() + 1e-6)
        return bce + (1 - dice)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/segmentation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_ds = PairedMaskDataset(args.data_dir, "train", args.image_size)
    val_ds = PairedMaskDataset(args.data_dir, "val", args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DiceBCELoss()
    best_dice = -1.0
    Path("weights").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        model.eval()
        scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                scores.append(dice_score(model(images), masks))
        val_dice = float(np.mean(scores)) if scores else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={running/max(len(train_ds),1):.4f} | val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "weights/mri_unet.pth")

    print("Best validation Dice:", best_dice)
    print("Evaluate on an untouched patient-level test set before reporting segmentation performance.")


if __name__ == "__main__":
    main()

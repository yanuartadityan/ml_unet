import torch
import torchvision
from dataset import CarvanaSet
from torch.utils.data import DataLoader


def save_chkpts(state, filename="chkpts.pth.tar"):
    print("::> saving checkpoint...")
    torch.save(state, filename)


def load_chkpts(checkpoint, model):
    print("::> loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])


def save_preds_as_imgs(loader, model, device, folder="saved_images/"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/y_{idx}.png")

    model.train()


def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    pin_memory,
):
    train_ds = CarvanaSet(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    val_ds = CarvanaSet(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_acc(loader, model, device):
    n_correct = 0
    n_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            n_correct += (preds == y).sum()
            n_pixels += torch.numel(preds)
            dice_score += 2 * (preds * y).sum() / ((preds + y).sum() + 1e-8)

    print(f"::Got {n_correct}/{n_pixels} with acc {n_correct/n_pixels}")
    print(f"::Dice score is {dice_score/len(loader)}")

    model.train()

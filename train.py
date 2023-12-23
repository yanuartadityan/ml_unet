import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNet
from utils import (
    get_loaders,
    save_chkpts,
    save_preds_as_imgs,
    check_acc,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 160
PIN_MEMORY = True
LOAD_MODEL = False
CROSS_ENTROPY_LOSS = False

TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward propagation.
        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)

        # backward propagation.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_tx = A.Compose(
        [
            A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_tx = A.Compose(
        [
            A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # create model instance.
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss() if CROSS_ENTROPY_LOSS else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_tx,
        val_tx,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    for epoch in range(NUM_EPOCHS):
        train_fn(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
        )

        # save.
        chkpts = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_chkpts(chkpts)

        # check acc.
        check_acc(
            val_loader,
            model,
            device=DEVICE,
        )

        # print.
        save_preds_as_imgs(val_loader, model, DEVICE)


if __name__ == "__main__":
    main()

import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import albumentations as A
from albumentations.pytorch import ToTensorV2

from os.path import join, splitext
from PIL import Image

import cv2


IMAGE_SIZE = 256
BASE_LR = 1e-4
BATCH_SIZE = 16
MAX_EPOCHS = 15

TRAIN_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

VAL_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

device = "gpu" if torch.cuda.is_available() else "cpu"


class BirdsDataset(Dataset):
    def __init__(self, dataset_path, mode, transform=None):
        self._items = []
        self._transform = transform

        img_dir_path = join(dataset_path, "images")
        mask_dir_path = join(dataset_path, "gt")

        if mode == "train":
            class_names = os.listdir(img_dir_path)[:-1]
        else:
            class_names = os.listdir(img_dir_path)[-2:-1]
        for class_name in class_names:
            for img_name in os.listdir(join(img_dir_path, class_name)):
                mask_name = splitext(img_name)[0] + ".png"
                self._items.append((
                    join(img_dir_path, class_name, img_name),
                    join(mask_dir_path, class_name, mask_name)
                ))

    def __getitem__(self, index: int):
        (img_path, mask_path) = self._items[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.asarray(image).astype(np.float32)
        mask = np.asarray(mask).astype(np.float32)
        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask[None, ...]

    def __len__(self):
        return len(self._items)


def conv_relu(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=False)  #ПОМЕНЯТЬ!
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = conv_relu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = conv_relu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = conv_relu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = conv_relu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = conv_relu(512, 512, 1, 0)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)

        self.conv_up3 = conv_relu(256 + 512, 512, 3, 1)
        self.conv_up2 = conv_relu(128 + 512, 256, 3, 1)
        self.conv_up1 = conv_relu(64 + 256, 256, 3, 1)
        self.conv_up0 = conv_relu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = conv_relu(3, 64, 3, 1)
        self.conv_original_size1 = conv_relu(64, 64, 3, 1)
        self.conv_original_size2 = conv_relu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.up_sample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.up_sample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.up_sample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.up_sample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.up_sample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +
                                                 target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class BirdsSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = ResNetUNet()

        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False

        self.bce_weight = 0.9

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        bce = F.binary_cross_entropy_with_logits(y_logit, y)

        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=BASE_LR,
                                     weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True)

        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }

        return [optimizer], [lr_dict]

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        bce = F.binary_cross_entropy_with_logits(y_logit, y)

        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (
                1 - self.bce_weight) * y.size(0)

        return {'val_loss': loss, 'logs': {'dice': dice, 'bce': bce}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print(f"| Train_loss: {avg_loss:.3f}")
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True,
                 on_step=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()

        print(
            f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}",
            end=" ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True,
                 on_step=False)


def train_segmentation_model(train_data_path):
    train_set = BirdsDataset(train_data_path, mode="train",
                             transform=TRAIN_TRANSFORM)
    val_set = BirdsDataset(train_data_path, mode="val",
                           transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    MyModelCheckpoint = ModelCheckpoint(dirpath='.',
                                        filename='{epoch}-{val_loss:.3f}',
                                        monitor='val_loss',
                                        mode='min',
                                        save_top_k=1)

    MyEarlyStopping = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=5,
                                    verbose=True)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=device,
        logger=False,
        callbacks=[MyEarlyStopping, MyModelCheckpoint]
    )

    model = BirdsSegmentation()
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), "segmentation_model.pth")


def get_model():
    return BirdsSegmentation()


def predict(model, img_path):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image = np.asarray(image).astype(np.float32)
    image_shape = image.shape[:-1]
    image = TRAIN_TRANSFORM(image=image)["image"][None, :]
    pred = torch.sigmoid(model.forward(image)).detach().numpy().astype(np.uint8)[0, 0]
    resize_transform = A.Resize(image_shape[0], image_shape[1], interpolation=cv2.INTER_LINEAR)
    pred = resize_transform(image=pred)["image"]
    return pred

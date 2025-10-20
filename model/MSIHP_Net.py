import math
from typing import List

import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class InceptionBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels_per_path):
        super(InceptionBlock1D, self).__init__()

        self.path_1x1 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=1)

        self.path_3x3 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=3, padding='same')

        self.path_5x5 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=5, padding='same')

        self.path_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels_per_path, kernel_size=1)
        )

    def forward(self, x):
        out_1x1 = self.path_1x1(x)
        out_3x3 = self.path_3x3(x)
        out_5x5 = self.path_5x5(x)
        out_pool = self.path_pool(x)

        x = torch.cat([out_1x1, out_3x3, out_5x5, out_pool], dim=1)

        return F.relu(x)


class HybridPooling(nn.Module):

    def __init__(self):
        super(HybridPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        return torch.cat([avg_out, max_out], dim=1)


class Inception_with_HybridPooling(nn.Module):

    def __init__(self, in_channel, num_classes, spectrum_length):
        super(Inception_with_HybridPooling, self).__init__()

        path_channels_1 = 8
        path_channels_2 = 16
        path_channels_3 = 32

        self.inception1 = InceptionBlock1D(in_channel, path_channels_1)
        self.inception2 = InceptionBlock1D(4 * path_channels_1, path_channels_2)
        self.inception3 = InceptionBlock1D(4 * path_channels_2, path_channels_3)

        self.pool = nn.MaxPool1d(2)

        self.global_pool = HybridPooling()

        final_inception_channels = 4 * path_channels_3  # -> 128

        classifier_input_dim = final_inception_channels * 2  # -> 256

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.inception1(x)
        x = self.pool(x)

        x = self.inception2(x)
        x = self.pool(x)

        x = self.inception3(x)
        x = self.pool(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


import torch
import torch.nn as nn
import pytorch_lightning as pl



class LitModelWithHistory(pl.LightningModule):

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def on_validation_epoch_end(self):

        train_loss = self.trainer.callback_metrics.get('train_loss')
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_acc = self.trainer.callback_metrics.get('val_acc')


        if train_loss is not None and val_loss is not None and val_acc is not None:

            self.history['train_loss'].append(train_loss.item())
            self.history['val_loss'].append(val_loss.item())
            self.history['val_acc'].append(val_acc.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

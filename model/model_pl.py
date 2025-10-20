import lightning.pytorch as pl
import torch
import torchmetrics
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch import optim
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

from .MSIHP_Net import Inception_with_HybridPooling


class BaselinePl(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = Inception_with_HybridPooling(
            in_channel=cfg["pc"]["in_channel"],
            num_classes=1,
            spectrum_length=cfg["spectrum_length"]
        )


        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.loss_fn = torch.nn.BCEWithLogitsLoss()


        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")


        self.test_label_pred = pd.DataFrame(columns=["label", "pred", "prob"])
        self.best_val_acc = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        spec, label = batch["spec"], batch["label"].float()
        logits = self(spec).squeeze(1)
        loss = self.loss_fn(logits, label)
        self.train_acc(logits, label.int())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        spec, label = batch["spec"], batch["label"].float()
        logits = self(spec).squeeze(1)
        loss = self.loss_fn(logits, label)
        self.val_acc(logits, label.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            acc = self.val_acc.compute()
            self.log("val_acc", acc, prog_bar=True)
            self.best_val_acc = max(self.best_val_acc, acc)
            self.log("best_val_acc", self.best_val_acc, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        spec, label = batch["spec"], batch["label"].float()
        logits = self(spec).squeeze(1)
        loss = self.loss_fn(logits, label)


        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.test_acc(logits, label.int())
        self.test_recall(logits, label.int())
        self.test_precision(logits, label.int())
        self.test_f1(logits, label.int())


        df = pd.DataFrame({
            "label": label.int().cpu().numpy(),
            "pred": preds.cpu().numpy(),
            "prob": probs.cpu().numpy()
        })
        self.test_label_pred = pd.concat([self.test_label_pred, df], ignore_index=True)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):

        self.log("test_acc", self.test_acc.compute())
        self.log("test_recall", self.test_recall.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_f1", self.test_f1.compute())


        self.plot_pr_curve()


        if self.cfg["log"]:
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.test_label_pred["label"].to_list(),
                preds=self.test_label_pred["pred"].to_list(),
                class_names=self.cfg["class_names"],
            )
            wandb.log({"confusion_matrix": cm})


    def plot_pr_curve(self):

        y_true = self.test_label_pred["label"].values
        y_scores = self.test_label_pred["prob"].values


        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)


        ap_score = average_precision_score(y_true, y_scores)
        pr_auc = auc(recall, precision)


        plt.figure(figsize=(8, 6))


        plt.plot(recall, precision, linewidth=2, color='blue',
                 label=f'PR curve (AP={ap_score:.4f}, AUC={pr_auc:.4f})')


        current_precision = self.test_precision.compute().cpu().item()
        current_recall = self.test_recall.compute().cpu().item()
        plt.scatter(current_recall, current_precision, s=150, c='red',
                    marker='*', zorder=5, edgecolors='black', linewidths=1.5,
                    label=f'Threshold=0.5\n(P={current_precision:.4f}, R={current_recall:.4f})')


        pos_ratio = np.sum(y_true) / len(y_true)
        plt.plot([0, 1], [pos_ratio, pos_ratio], 'k--',
                 linewidth=1.5, alpha=0.7,
                 label=f'Random classifier (baseline={pos_ratio:.4f})')


        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curve for Green Pea Galaxy Classification',
                  fontsize=14, fontweight='bold', pad=15)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])


        plt.tight_layout()

        save_path = 'pr_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if self.cfg.get("log", False):
            wandb.log({
                "pr_curve_image": wandb.Image(plt),
                "average_precision": ap_score,
                "pr_auc": pr_auc
            })

        plt.close()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.cfg["lr"])
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.cfg["T_0"], T_mult=self.cfg["T_mult"], eta_min=self.cfg["eta_min"]
        )
        return [opt], [{"scheduler": sch, "interval": "epoch", "name": "lr"}]

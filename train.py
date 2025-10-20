import os
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    Callback,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    PrecisionRecallDisplay,
)

from cfg.cfg import cfg
from dataset.spectrum import SpectrumDataset
from model_new.model_pl import BaselinePl
from utils.tools import seed_every_thing, get_free_gpu_index, arg_parser


class MetricsCallback(Callback):
    """收集每个 epoch 的 train_loss、val_loss、val_acc"""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        vl = metrics.get("val_loss")
        va = metrics.get("val_acc")
        if vl is not None and va is not None:
            self.val_losses.append(vl.cpu().item())
            self.val_accs.append(va.cpu().item())


def main():
    plt.rcParams['font.family'] = 'Times New Roman'

    # seed & device
    seed_every_thing(cfg["seed"])
    if cfg["device"] == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            get_free_gpu_index(cfg["device_list"])
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["device"])

    # datasets
    train_ds = SpectrumDataset(os.path.join(cfg["data_dir"], "train"), cfg["spectrum_length"], cfg["class_names"])
    val_ds = SpectrumDataset(os.path.join(cfg["data_dir"], "val"), cfg["spectrum_length"], cfg["class_names"])
    test_ds = SpectrumDataset(os.path.join(cfg["data_dir"], "test"), cfg["spectrum_length"], cfg["class_names"])

    # dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # model
    model = BaselinePl(cfg)

    # logger & callbacks
    loggers = []
    callbacks = [ModelSummary(max_depth=-1)]
    metrics_cb = MetricsCallback()

    if cfg["log"]:
        tb = TensorBoardLogger(save_dir=cfg["log_path"], name=cfg["sweep"])
        wandb_logger = WandbLogger(project=cfg["project"], name=cfg["sweep"])
        loggers += [tb, wandb_logger]

        callbacks += [
            ModelCheckpoint(
                monitor="val_acc", save_top_k=3, mode="max"
            ),
            EarlyStopping(
                monitor="val_acc",
                patience=cfg["patience"],
                min_delta=cfg["min_delta"],
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]


    callbacks.append(metrics_cb)

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        precision=cfg["precision"],
        fast_dev_run=cfg["debug"],
        logger=loggers or None,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dl, val_dl)

    epochs_train = range(1, len(metrics_cb.train_losses) + 1)
    epochs_val = range(1, len(metrics_cb.val_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_train, metrics_cb.train_losses, label="train_loss")
    plt.plot(epochs_val, metrics_cb.val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_val, metrics_cb.val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_acc_curve.png")
    plt.close()

    if cfg["test"]:
        ckpt = None
        if cfg["log"]:
            # callbacks[1] 是 ModelCheckpoint
            ckpt = callbacks[1].best_model_path
        trainer.test(model, test_dl, ckpt_path=ckpt)

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_dl:
                specs = batch["spec"].to(model.device)
                labels = batch["label"].to(model.device)
                logits = model(specs).squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())


        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_disp.plot()

        plt.tight_layout()
        plt.savefig("pr_curve.png", dpi=300, bbox_inches='tight')
        plt.savefig("pr_curve.pdf", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    arg_parser(cfg)
    main()

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        spectrum_length: int,
        class_names=("green_pea", "non_green_pea"),
        spectrum_suffix: str = ".csv",
    ) -> None:
        super().__init__()
        self.spectrum_dir = os.path.join(data_dir, "spectrum")
        self.label_dir = os.path.join(data_dir, "label")
        self.spectrum_length = spectrum_length
        if isinstance(class_names, str):
            class_names = [c.strip() for c in class_names.split(",")]
        self.class_names = list(class_names)
        self.suffix = spectrum_suffix

        self.label_df = pd.read_csv(os.path.join(self.label_dir, "label.csv"))

        unknown = set(self.label_df["label"]) - set(self.class_names)
        if unknown:
            print(f"Warning: filtering out unknown labels {unknown}")
            self.label_df = self.label_df[self.label_df["label"].isin(self.class_names)]
        self.label_df = self.label_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.label_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.label_df.iloc[idx]
        basename = row["basename"]
        label_str = row["label"]
        # 1. 读文件（NumPy array）
        path = os.path.join(self.spectrum_dir, basename + self.suffix)
        if self.suffix == ".csv":
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        else:
            arr = np.load(path)
        intensity = arr[:, 1]

        L = intensity.shape[0]
        if L < self.spectrum_length:
            pad_left = (self.spectrum_length - L) // 2
            pad_right = self.spectrum_length - L - pad_left
            intensity = np.pad(intensity, (pad_left, pad_right), mode="constant", constant_values=0)
        else:
            intensity = intensity[: self.spectrum_length]


        spec = torch.from_numpy(intensity).float().unsqueeze(0)  # [1, length]

        label_idx = self.class_names.index(label_str)
        label = torch.tensor(label_idx, dtype=torch.long)

        return {"spec": spec, "label": label}

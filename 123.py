#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MNIST 手写数字识别：单文件完整实现（纯 PyTorch，不依赖第三方数据集封装）。

作业要求对应关系：
1. 不依赖第三方数据集封装：本文件自己解析官方 IDX 二进制文件；
2. 包含模型结构、训练过程、测试评估、曲线和混淆矩阵；
3. 所有代码集中在本文件，直接运行 `python 123.py` 即可。

常用命令：
    python 123.py                       # 默认训练 10 轮
    python 123.py --epochs 1             # 快速跑 1 轮
    python 123.py --device cpu           # 没有 GPU 时
    python 123.py --no-amp --num-workers 0
    python 123.py --download             # 数据缺失时自动下载

数据目录结构（4 个 .gz 文件）：
    data/MNIST/raw/train-images-idx3-ubyte.gz
    data/MNIST/raw/train-labels-idx1-ubyte.gz
    data/MNIST/raw/t10k-images-idx3-ubyte.gz
    data/MNIST/raw/t10k-labels-idx1-ubyte.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # 无显示器环境也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

# =============================================================================
# 0. 常量与路径
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_SUBDIR = Path("MNIST") / "raw"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CLASS_NAMES = [str(i) for i in range(10)]

# 文件名 -> (文件名, 官方 MD5)。下载后必须校验。
MNIST_FILES: dict[str, tuple[str, str]] = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
)


# =============================================================================
# 1. 数据下载与 IDX 解析
# =============================================================================


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_idx_images(path: str | Path) -> np.ndarray:
    """读取 IDX3 图像文件，返回 uint8 数组 (N, H, W)。

    IDX 头部 16 字节：magic=2051, N, H, W，均为大端序。
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()

    if len(raw) < 16:
        raise ValueError(f"文件过短，不是合法的 IDX 图像文件: {path}")
    magic, num, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"magic number 应为 2051，实际为 {magic}: {path}")

    expected = 16 + num * rows * cols
    if len(raw) != expected:
        raise ValueError(
            f"文件长度不符：期望 {expected} 字节，实际 {len(raw)} 字节: {path}"
        )
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(num, rows, cols)


def read_idx_labels(path: str | Path) -> np.ndarray:
    """读取 IDX1 标签文件，返回 uint8 数组 (N,)。"""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()

    if len(raw) < 8:
        raise ValueError(f"文件过短，不是合法的 IDX 标签文件: {path}")
    magic, num = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"magic number 应为 2049，实际为 {magic}: {path}")

    expected = 8 + num
    if len(raw) != expected:
        raise ValueError(
            f"文件长度不符：期望 {expected} 字节，实际 {len(raw)} 字节: {path}"
        )
    return np.frombuffer(raw, dtype=np.uint8, offset=8).copy()


def _download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mnist-cnn/1.0"})

    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with part.open("wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / total * 100:5.1f}%", end="")
        print()
    part.replace(dest)


def download_mnist(data_dir: Path, force: bool = False) -> Path:
    """确保四个文件存在且 MD5 正确；缺失/损坏时才联网下载。"""
    raw_dir = data_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_md5 in MNIST_FILES.values():
        dest = raw_dir / filename
        if dest.exists() and not force:
            if _md5(dest) == expected_md5:
                print(f"[数据] 已存在且校验通过: {filename}")
                continue
            print(f"[数据] MD5 不匹配，重新下载: {filename}")

        errors: list[str] = []
        for mirror in MIRRORS:
            url = f"{mirror.rstrip('/')}/{filename}"
            try:
                print(f"[数据] 下载 {url}")
                _download_one(url, dest)
                actual = _md5(dest)
                if actual != expected_md5:
                    dest.unlink(missing_ok=True)
                    raise ValueError(f"MD5 校验失败：期望 {expected_md5}，实际 {actual}")
                print(f"[数据] 校验通过: {filename}")
                break
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                errors.append(f"{url} -> {exc}")
                print(f"[数据] 镜像失败: {exc}")
        else:
            detail = "\n".join(f"  - {item}" for item in errors)
            raise RuntimeError(
                f"无法下载 {filename}。请检查网络，或手动放到 {raw_dir}。\n{detail}"
            )
    return raw_dir


# =============================================================================
# 2. Dataset、Transform 与数据划分
# =============================================================================


class MNISTDataset(Dataset):
    """内存版 MNIST，返回 (image, label)。

    图像为 float32、范围 [0,1]、形状 (1,28,28)。标准化和增强由 transform 完成，
    这样训练集和验证集可以共用原始数据，但使用不同预处理。
    """

    def __init__(
        self,
        data_dir: Path,
        train: bool = True,
        download: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / RAW_SUBDIR
        self.train = train
        self.transform = transform

        if download:
            download_mnist(self.data_dir)

        prefix = "train" if train else "t10k"
        image_path = self.raw_dir / f"{prefix}-images-idx3-ubyte.gz"
        label_path = self.raw_dir / f"{prefix}-labels-idx1-ubyte.gz"

        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"未找到数据文件: {image_path} 或 {label_path}\n"
                f"请运行 `python 123.py --download`，"
                f"或手动把 4 个 .gz 文件放到 {self.raw_dir}/"
            )

        images = read_idx_images(image_path)
        labels = read_idx_labels(label_path)
        if len(images) != len(labels):
            raise ValueError(f"图像数 {len(images)} 与标签数 {len(labels)} 不一致")

        # 一次性读入内存：60k 张图约 47MB，避免每个 epoch 反复解压。
        # copy 一次是为了绕开 np.frombuffer 返回只读数组的警告。
        self.data = (
            torch.from_numpy(images.copy())
            .float()
            .div_(255.0)
            .unsqueeze_(1)
            .contiguous()
        )
        self.targets = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])


class Normalize:
    """(x - mean) / std。"""

    def __init__(self, mean: float = MNIST_MEAN, std: float = MNIST_STD) -> None:
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class RandomShift:
    """随机平移：补零后随机裁剪回 28x28，幅度不超过 max_shift 像素。"""

    def __init__(self, max_shift: int = 2) -> None:
        if max_shift < 0:
            raise ValueError("max_shift 不能为负数")
        self.max_shift = int(max_shift)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError("RandomShift 期望单张图像，形状应为 (C, H, W)")
        if self.max_shift == 0:
            return image
        pad = self.max_shift
        height, width = image.shape[-2:]
        # 先加 batch 维再 pad，确保 4 元组同时作用在 H 和 W 两个方向。
        padded = torch.nn.functional.pad(
            image.unsqueeze(0), (pad, pad, pad, pad), mode="constant", value=0.0
        ).squeeze(0)
        top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        return padded[:, top : top + height, left : left + width]

    def __repr__(self) -> str:
        return f"RandomShift(max_shift={self.max_shift})"


class Compose:
    def __init__(self, *transforms: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image

    def __repr__(self) -> str:
        return "Compose(" + ", ".join(repr(t) for t in self.transforms) + ")"


class TransformSubset(Dataset):
    """给 Subset 单独套 transform，防止验证集被数据增强污染。"""

    def __init__(
        self,
        subset: Subset,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_split(
    dataset: MNISTDataset, val_size: int = 6000, seed: int = 42
) -> tuple[Subset, Subset]:
    """按类别分层切分训练集，保证验证集里 0~9 的比例均衡。"""
    if not 0 < val_size < len(dataset):
        raise ValueError(f"val_size 必须在 (0, {len(dataset)}) 内，实际为 {val_size}")

    targets = dataset.targets.numpy()
    rng = np.random.default_rng(seed)
    per_class = val_size // 10
    remainder = val_size % 10

    val_indices: list[int] = []
    for class_id in range(10):
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        take = per_class + (1 if class_id < remainder else 0)
        val_indices.extend(indices[:take].tolist())

    val_set = set(val_indices)
    train_indices = [i for i in range(len(dataset)) if i not in val_set]
    assert not (set(train_indices) & val_set)
    assert len(train_indices) + len(val_indices) == len(dataset)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# =============================================================================
# 3. 模型
# =============================================================================


class MNISTCNN(nn.Module):
    """小型 CNN：4 个 3x3 卷积 + 2 次池化 + 全连接分类头。

    输入 (B,1,28,28) -> 32x28x28 -> 32x14x14 -> 64x14x14 -> 64x7x7
         -> Flatten 3136 -> Linear 128 -> Linear 10
    """

    def __init__(
        self, num_classes: int = 10, dropout: float = 0.25, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 至少为 2")

        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# =============================================================================
# 4. 训练 / 评估
# =============================================================================


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
    max_batches: int = 0,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.detach().item(), batch_size)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return {
        "loss": loss_meter.avg,
        "accuracy": correct / total if total else 0.0,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
    collect_predictions: bool = False,
    max_batches: int = 0,
) -> dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        correct += (predictions == targets).sum().item()
        total += batch_size

        if collect_predictions:
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            flat_index = targets.cpu() * num_classes + predictions.cpu()
            confusion += torch.bincount(
                flat_index, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    result: dict[str, object] = {
        "loss": loss_meter.avg,
        "accuracy": correct / total if total else 0.0,
        "num_samples": total,
    }
    if collect_predictions:
        result["confusion_matrix"] = confusion
        result["targets"] = all_targets
        result["predictions"] = all_predictions
    return result


def confusion_metrics(confusion: torch.Tensor) -> dict[str, object]:
    """由混淆矩阵计算每类准确率、精确率、召回率、F1。"""
    matrix = confusion.to(torch.float64)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diag()

    per_class_accuracy = torch.where(
        support > 0, true_positive / support, torch.zeros_like(support)
    )
    precision = torch.where(
        predicted > 0, true_positive / predicted, torch.zeros_like(predicted)
    )
    recall = per_class_accuracy
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    return {
        "per_class_accuracy": [round(v, 6) for v in per_class_accuracy.tolist()],
        "precision": [round(v, 6) for v in precision.tolist()],
        "recall": [round(v, 6) for v in recall.tolist()],
        "f1": [round(v, 6) for v in f1.tolist()],
        "macro_f1": round(float(f1.mean().item()), 6),
    }


# =============================================================================
# 5. 工具与可视化
# =============================================================================


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 检测不到可用 GPU")
    return device


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: object, path: Path) -> None:
    def default(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"无法序列化类型: {type(obj)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=default) + "\n",
        encoding="utf-8",
    )


def plot_history(
    history: dict[str, list[float]], path: Path, test_accuracy: float | None = None
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["train_loss"], "o-", label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], "s-", label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss curves")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    # history 里存的是 0~1 的小数，画图时转换成百分比，避免纵轴看起来“贴地”。
    train_acc_pct = [value * 100 for value in history["train_acc"]]
    val_acc_pct = [value * 100 for value in history["val_acc"]]
    ax_acc.plot(epochs, train_acc_pct, "o-", label="Train accuracy")
    ax_acc.plot(epochs, val_acc_pct, "s-", label="Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy curves")
    ax_acc.grid(alpha=0.3)
    if test_accuracy is not None:
        ax_acc.axhline(
            test_accuracy * 100,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Test {test_accuracy * 100:.2f}%",
        )
    ax_acc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_samples(dataset: MNISTDataset, path: Path, num_samples: int = 32) -> None:
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    columns = 8
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.7))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, index in zip(axes_array, indices):
        image, label = dataset[int(index)]
        ax.imshow(image.squeeze(0).numpy(), cmap="gray")
        ax.set_title(f"label = {label}", fontsize=9)
        ax.axis("off")
    for ax in axes_array[len(indices) :]:
        ax.axis("off")

    fig.suptitle("Random MNIST test samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion: torch.Tensor,
    path: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    matrix = confusion.numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix on test set")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if value:
                ax.text(
                    col,
                    row,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > threshold else "black",
                )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    keys = ["train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for index in range(len(history["train_loss"])):
            writer.writerow([index + 1] + [history[key][index] for key in keys])


# =============================================================================
# 6. 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST 手写数字识别（单文件，纯 PyTorch）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=6000)
    parser.add_argument("--random-shift", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0 / mps")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None,
        help="混合精度；默认 CUDA 上开启，可用 --no-amp 关闭",
    )
    parser.add_argument("--download", action="store_true", help="数据缺失时自动下载")
    parser.add_argument(
        "--download-only", action="store_true", help="只下载/校验数据，然后退出"
    )
    parser.add_argument(
        "--limit-batches", type=int, default=0,
        help="每个阶段最多跑多少个 batch，仅用于冒烟测试；0 表示不限制",
    )
    return parser.parse_args()


def build_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs 必须 >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须 >= 1")
    if args.limit_batches < 0:
        raise SystemExit("--limit-batches 必须 >= 0")
    if args.num_workers < 0:
        raise SystemExit("--num-workers 必须 >= 0")

    if args.download_only:
        raw_dir = download_mnist(args.data_dir)
        print(f"\n数据已就绪: {raw_dir}")
        return

    device = get_device(args.device)
    use_amp = (device.type == "cuda") if args.amp is None else args.amp
    if use_amp and device.type != "cuda":
        use_amp = False
        print("[提示] 只有 CUDA 支持当前混合精度设置，已自动关闭 AMP。")

    seed_everything(args.seed, deterministic=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MNIST CNN 训练")
    print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | 设备 {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("=" * 78)

    # 1. 数据
    print("\n[1/5] 加载数据 ...")
    train_base = MNISTDataset(args.data_dir, train=True, download=args.download)
    test_base = MNISTDataset(args.data_dir, train=False, download=args.download)
    train_subset, val_subset = stratified_split(
        train_base, val_size=args.val_size, seed=args.seed
    )

    normalize = Normalize(MNIST_MEAN, MNIST_STD)
    train_set = TransformSubset(
        train_subset, Compose(RandomShift(args.random_shift), normalize)
    )
    val_set = TransformSubset(val_subset, normalize)
    test_set = TransformSubset(test_base, normalize)

    print(
        f"  训练集 {len(train_set):5d} | 验证集 {len(val_set):5d} | "
        f"测试集 {len(test_set):5d}"
    )
    print(f"  训练 transform: {train_set.transform}")
    print(f"  验证/测试 transform: {normalize}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = build_loader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        generator=generator,
        drop_last=len(train_set) >= args.batch_size,
    )
    val_loader = build_loader(
        val_set,
        args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = build_loader(
        test_set,
        args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    # 2. 模型
    print("\n[2/5] 构建模型 ...")
    model = MNISTCNN(
        num_classes=10, dropout=args.dropout, hidden_dim=args.hidden_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"  可训练参数: {count_parameters(model):,}")
    print(f"  损失函数: CrossEntropyLoss(label_smoothing={args.label_smoothing})")
    print(f"  优化器: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  混合精度: {'开启' if use_amp else '关闭'}")

    model_config = {
        "num_classes": 10,
        "dropout": args.dropout,
        "hidden_dim": args.hidden_dim,
    }
    best_path = args.output_dir / "best_model.pt"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "seconds": [],
    }
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    # 3. 训练
    print("\n[3/5] 开始训练 ...")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            grad_clip=args.grad_clip,
            max_batches=args.limit_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.limit_batches,
        )
        elapsed = time.perf_counter() - started
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(round(train_metrics["loss"], 6))
        history["train_acc"].append(round(train_metrics["accuracy"], 6))
        history["val_loss"].append(round(val_metrics["loss"], 6))
        history["val_acc"].append(round(val_metrics["accuracy"], 6))
        history["lr"].append(current_lr)
        history["seconds"].append(round(elapsed, 3))

        improved = val_metrics["accuracy"] > best_val_acc + 1e-9 or (
            abs(val_metrics["accuracy"] - best_val_acc) <= 1e-9
            and val_metrics["loss"] < best_val_loss
        )
        marker = ""
        if improved:
            best_val_acc = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "normalization": [MNIST_MEAN, MNIST_STD],
                    "epoch": epoch,
                    "val_accuracy": best_val_acc,
                    "val_loss": best_val_loss,
                    "seed": args.seed,
                },
                best_path,
            )
            marker = "  <-- best"

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy'] * 100:5.2f}% | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy'] * 100:5.2f}% | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s{marker}",
            flush=True,
        )

        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"\n验证集已连续 {args.patience} 轮没有提升，提前停止。")
            break

    print(f"\n训练结束：最佳验证集准确率 {best_val_acc * 100:.2f}%（第 {best_epoch} 轮）")

    # 4. 测试
    print("\n[4/5] 在测试集上评估最佳模型 ...")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_classes=10,
        collect_predictions=True,
        max_batches=args.limit_batches,
    )
    confusion = test_metrics.pop("confusion_matrix")
    assert isinstance(confusion, torch.Tensor)
    test_metrics.pop("targets", None)
    test_metrics.pop("predictions", None)

    detailed = confusion_metrics(confusion)
    test_metrics.update(detailed)
    test_metrics["confusion_matrix"] = confusion.tolist()

    print(
        f"  测试集 loss {test_metrics['loss']:.4f} | "
        f"accuracy {test_metrics['accuracy'] * 100:.2f}% | "
        f"macro-F1 {detailed['macro_f1']:.4f}"
    )
    per_class = ", ".join(
        f"{index}:{value * 100:.2f}%"
        for index, value in enumerate(detailed["per_class_accuracy"])
    )
    print(f"  每类准确率: {per_class}")

    # 5. 保存产物
    print("\n[5/5] 保存结果 ...")
    plot_history(history, args.output_dir / "training_curves.png", test_metrics["accuracy"])
    plot_samples(test_base, args.output_dir / "samples.png", num_samples=32)
    plot_confusion_matrix(confusion, args.output_dir / "confusion_matrix.png")
    save_history_csv(history, args.output_dir / "history.csv")

    checkpoint.update(
        {
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "macro_f1": detailed["macro_f1"],
            "trained_epochs": len(history["train_loss"]),
            "class_names": CLASS_NAMES,
        }
    )
    torch.save(checkpoint, best_path)

    metrics = {
        "dataset": {
            "name": "MNIST",
            "train_size": len(train_set),
            "val_size": len(val_set),
            "test_size": len(test_set),
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
        },
        "model": {**model_config, "parameters": count_parameters(model)},
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": args.label_smoothing,
            "random_shift": args.random_shift,
            "amp": use_amp,
            "seed": args.seed,
        },
        "history": history,
        "test": test_metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(),
        },
    }
    save_json(metrics, args.output_dir / "metrics.json")
    save_json(vars(args), args.output_dir / "run_config.json")

    print("\n生成的文件：")
    for name in [
        "best_model.pt",
        "history.csv",
        "metrics.json",
        "run_config.json",
        "training_curves.png",
        "samples.png",
        "confusion_matrix.png",
    ]:
        path = args.output_dir / name
        print(f"  {path}  ({path.stat().st_size / 1024:.1f} KiB)")
    print(f"\n完成！测试集准确率 {test_metrics['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MultiLabelImageDataset(Dataset):
    """Reads (image, multi-hot label vector) pairs from a flat image directory
    (<image_dir>/<split>/<filename>) plus a manifest CSV with one row per
    image and one 0/1 column per entry in label_columns -- for tasks like
    biomarker classification where labels aren't mutually exclusive, so
    torchvision's ImageFolder (single label per image) doesn't fit."""

    def __init__(self, labels_csv, image_dir, split, label_columns, transform=None):
        df = pd.read_csv(labels_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows with split={split!r} in {labels_csv}")

        self.image_dir = os.path.join(image_dir, split)
        self.filenames = df["filename"].tolist()
        self.labels = df[label_columns].to_numpy(dtype="float32")
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.filenames[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(self.labels[idx])


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if getattr(args, "multilabel", False):
        label_columns = [c.strip() for c in args.label_columns.split(",") if c.strip()]
        dataset = MultiLabelImageDataset(
            labels_csv=args.labels_csv, image_dir=args.data_path,
            split=is_train, label_columns=label_columns, transform=transform,
        )
        if is_train == 'train':
            dataset = _maybe_subsample_train(dataset, args)
        return dataset

    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)
    if is_train == 'train':
        dataset = _maybe_subsample_train(dataset, args, stratified_targets=dataset.targets)
    return dataset


def _maybe_subsample_train(dataset, args, stratified_targets=None):
    ratio = float(getattr(args, "dataratio", 1.0))
    seed = int(getattr(args, "seed", 0))
    stratified = bool(getattr(args, "stratified", False))

    if not (0.0 < ratio < 1.0):
        return dataset

    if stratified:
        if stratified_targets is None:
            raise ValueError("--stratified requires single-label integer targets; "
                              "not supported together with --multilabel.")
        idx = _stratified_indices(stratified_targets, ratio, seed)
    else:
        # simple uniform subsample with torch.Generator for reproducibility
        g = torch.Generator().manual_seed(seed)
        n = len(dataset)
        k = max(1, int(n * ratio))
        idx = torch.randperm(n, generator=g)[:k].tolist()

    return Subset(dataset, idx)

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)

# ---- helpers ----

def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain class proportions. Ensures at least 1 sample per class when possible."""
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # shuffle final indices (stable across seed)
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep


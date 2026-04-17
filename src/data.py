from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import config
from pathlib import Path


def _resolve_data_strategy(model_name=None):
    defaults = {
        "augmentation_enabled": getattr(config, "AUGMENTATION_ENABLED", True),
        "aug_rotation_degrees": float(getattr(config, "AUG_ROTATION_DEGREES", 5.0)),
        "aug_affine_translate": float(getattr(config, "AUG_AFFINE_TRANSLATE", 0.0)),
        "aug_affine_scale_delta": float(getattr(config, "AUG_AFFINE_SCALE_DELTA", 0.0)),
        "aug_brightness": float(getattr(config, "AUG_BRIGHTNESS", 0.0)),
        "aug_contrast": float(getattr(config, "AUG_CONTRAST", 0.0)),
    }
    return config.resolve_model_strategy(defaults, model_name=model_name)

def load_metadata(csv_path, image_root):
    df = pd.read_csv(csv_path)
    
    image_root = Path(image_root)
    all_images = list(image_root.rglob("*.png"))
    image_map = {p.name: str(p) for p in all_images}
    
    df["image_path"] = df["Image Index"].map(image_map)
    
    return df
    # Load the metadata from a CSV file, creating a new column "image_path" that maps image filenames to their full paths on disk (handles subfolders via rglob)

def split_by_patient(df):
    patients = df["Patient ID"].unique()
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(patients)

    n = len(patients)
    train_ids = set(patients[:int(config.TRAIN_SPLIT * n)])
    val_ids = set(patients[int(config.TRAIN_SPLIT * n):int((config.TRAIN_SPLIT + config.VAL_SPLIT) * n)])
    test_ids = set(patients[int((config.TRAIN_SPLIT + config.VAL_SPLIT) * n):])

    def assign_split(pid):
        if pid in train_ids:
            return "train"
        elif pid in val_ids:
            return "val"
        return "test"
    
    df = df.copy()
    df["split"] = df["Patient ID"].apply(assign_split)

    return df
    # Split the dataframe into training, validation, and test sets based on unique patient IDs to

def create_binary_target(df):
    df = df.copy()
    df["target"] = (df["Finding Labels"] != "No Finding").astype(int)
    proportions = df["target"].value_counts(normalize=True)
    
    return df, proportions

def prepare_full_dataframe(csv_path, image_root):
    df = load_metadata(csv_path, image_root)
    df = split_by_patient(df)
    df, _ = create_binary_target(df)

    return df
    # Main function to prepare the full dataframe: load the metadata and then split it by patient to create the "split" column for later use in creating datasets and dataloaders

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        label = row["target"]

        if self.transform:
            image = self.transform(image)

        return image, label
    # Custom dataset class to load images and labels, applying transformations as needed

def get_transforms(model_name=None):
    strategy = _resolve_data_strategy(model_name)

    train_ops = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]

    if bool(strategy.get("augmentation_enabled", True)):
        rotation_degrees = max(0.0, float(strategy.get("aug_rotation_degrees", 0.0)))
        if rotation_degrees > 0.0:
            train_ops.append(transforms.RandomRotation(rotation_degrees))

        affine_translate = max(0.0, float(strategy.get("aug_affine_translate", 0.0)))
        affine_scale_delta = max(0.0, float(strategy.get("aug_affine_scale_delta", 0.0)))
        if affine_translate > 0.0 or affine_scale_delta > 0.0:
            scale_range = None
            if affine_scale_delta > 0.0:
                scale_range = (max(0.1, 1.0 - affine_scale_delta), 1.0 + affine_scale_delta)
            train_ops.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(affine_translate, affine_translate) if affine_translate > 0.0 else None,
                    scale=scale_range,
                )
            )

        aug_brightness = max(0.0, float(strategy.get("aug_brightness", 0.0)))
        aug_contrast = max(0.0, float(strategy.get("aug_contrast", 0.0)))
        if aug_brightness > 0.0 or aug_contrast > 0.0:
            train_ops.append(
                transforms.ColorJitter(
                    brightness=aug_brightness,
                    contrast=aug_contrast,
                )
            )

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    train_transforms = transforms.Compose(train_ops)
    # Augmentation for training set: resize + optional configurable transforms + normalize.

    eval_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    # For validation and test sets: only resize, convert to tensor and normalize, no augmentation

    return train_transforms, eval_transforms


def prepare_data(df, model_name=None):
    train_transforms, eval_transforms = get_transforms(model_name=model_name)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_ds = ChestXrayDataset(train_df, transform=train_transforms)
    val_ds = ChestXrayDataset(val_df, transform=eval_transforms)
    test_ds = ChestXrayDataset(test_df, transform=eval_transforms)

    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def sample_image_path(df, split="test", seed=config.RANDOM_SEED):
    df_split = df[df["split"] == split]
    return df_split.sample(1, random_state=seed)["image_path"].iloc[0]

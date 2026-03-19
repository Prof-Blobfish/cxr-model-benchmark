from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import config
from pathlib import Path

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

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    # Augmentation for training set: resize, random rotate, convert to tensore and normalize

    eval_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    # For validation and test sets: only resize, convert to tensor and normalize, no augmentation

    return train_transforms, eval_transforms


def split_dataframe(df):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    return train_df, val_df, test_df
    # Split the dataframe into training, validation, and test sets based on the "split" column

def create_datasets(train_df, val_df, test_df, train_transforms, eval_transforms):
    train_ds = ChestXrayDataset(train_df, transform = train_transforms)
    val_ds = ChestXrayDataset(val_df, transform = eval_transforms)
    test_ds = ChestXrayDataset(test_df, transform = eval_transforms)

    return train_ds, val_ds, test_ds
    # Create dataset instances for each split, applying the appropriate transformations

def create_dataloaders(train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size = config.BATCH_SIZE, shuffle = False, num_workers = config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size = config.BATCH_SIZE, shuffle = False, num_workers = config.NUM_WORKERS)
    
    return train_loader, val_loader, test_loader
    # Create dataloaders for each split, with shuffling for training and no shuffling for validation and test sets

def prepare_data(df):
    train_transforms, eval_transforms = get_transforms()
    train_df, val_df, test_df = split_dataframe(df)
    train_ds, val_ds, test_ds = create_datasets(train_df, val_df, test_df, train_transforms, eval_transforms)
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds)

    return train_loader, val_loader, test_loader
    # Main function to prepare the data: get transformations, split the dataframe, create datasets and dataloaders, and return the dataloaders for training, validation, and testing

def sample_image_path(df, split="test", seed=config.RANDOM_SEED):
    df_split = df[df["split"] == split]
    return df_split.sample(1, random_state=seed)["image_path"].iloc[0]

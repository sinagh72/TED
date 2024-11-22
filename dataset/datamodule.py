# --------------------------------------------------------
# datamodule to load train, val and test data
# Written by Sina Gholami
# --------------------------------------------------------
import os.path
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning.pytorch as pl
import torch


class CustomDataset(Dataset):

    def __init__(self, transform=None, dataset=None, img_type="RGB"):
        """
        hugging face login
        """
        self.dataset = dataset
        self.transform = transform
        self.img_type = img_type

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img_path = img_path.replace("\\", "/")  # fixing the path for windows os
        img_view = self.load_img(img_path)  # return an image
        if self.transform is not None:
            img_view = self.transform(img_view)
        return img_view, label

    def __len__(self):
        return len(self.dataset)

    def load_img(self, img_path):
        img = Image.open(img_path).convert(self.img_type)
        return img


class CustomDatamodule(pl.LightningDataModule):
    def __init__(self, root_dir, dataset_name=None, batch_size=None, train_transform=None, test_transform=None,
                 classes={}):
        """
        :param batch_size: int
        :param train_transform: transforms
        :param test_transform: transforms
        :param classes: dictionary of classes: NORMAL: 0
        """
        super().__init__()
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.classes = classes

    def prepare_data(self) -> None:
        self.train_data = load_from_csv(self.root_dir, "train.csv")
        self.val_data = load_from_csv(self.root_dir, "val.csv")
        self.test_data = load_from_csv(self.root_dir, "test.csv")

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = CustomDataset(transform=self.train_transform, dataset=self.train_data)
            print(f"{self.dataset_name} train data len:", len(self.data_train))
            print(self.train_data[0])
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = CustomDataset(transform=self.test_transform, dataset=self.val_data)
            print(f"{self.dataset_name} val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = CustomDataset(transform=self.test_transform, dataset=self.test_data)
            print(f"{self.dataset_name} test data len:", len(self.data_test))

    def train_dataloader(self, shuffle: bool = True, drop_last: bool = True, pin_memory: bool = True,
                         workers: int = torch.cuda.device_count() * 2):
        """
        :param num_workers: int, number of workers for training loder training
        """
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          pin_memory=pin_memory,
                          drop_last=drop_last,
                          num_workers=workers,
                          persistent_workers=True)

    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = True,
                       workers: int = torch.cuda.device_count() * 2):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=workers,
                          pin_memory=pin_memory,
                          persistent_workers=True)

    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = True,
                        workers: int = torch.cuda.device_count() * 2):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=workers,
                          pin_memory=pin_memory,
                          persistent_workers=True)


def load_from_csv(root_dir, csv_file):
    data_df = pd.read_csv(os.path.join(root_dir, csv_file))
    return [(row['Directory'], row['Label']) for _, row in data_df.iterrows()]


# def load_data(root, train_percentage=0.8):
#     train_df = pd.read_csv(os.path.join(root, "peopleDevTrain.csv"))
#     # Calculate the total number of images
#     total_images = train_df['images'].sum()
#
#     # Calculate the split index
#     split_index = (train_df['cumulative_images'] <= (train_percentage * total_images)).sum()
#
#     # Split into train and validation
#     train_split = train_df.iloc[:split_index]
#     val_split = train_df.iloc[split_index:]
#
#     train_images = []
#     val_images = []
#
#     for _, row in train_split.iterrows():
#         train_dir = os.path.join(root, "lfw-deepfunneled", "lfw-deepfunneled", row['name'])
#         if os.path.isdir(train_dir):
#             for img in os.listdir(train_dir):
#                 train_images.append(os.path.join(train_dir, img))
#         else:
#             print(f"Directory {train_dir} does not exist.")
#
#     # Load images for validation
#     for _, row in val_split.iterrows():
#         val_dir = os.path.join(root, "lfw-deepfunneled", "lfw-deepfunneled", row['name'])
#         if os.path.isdir(val_dir):
#             for img in os.listdir(val_dir):
#                 val_images.append(os.path.join(val_dir, img))
#         else:
#             print(f"Directory {val_dir} does not exist.")
#
#     test_df = pd.read_csv(os.path.join(root, "peopleDevTest.csv"))
#     test_images = []
#
#     for name in test_df['name']:
#         test_dir = os.path.join(root, "lfw-deepfunneled", "lfw-deepfunneled", name)
#         if os.path.isdir(test_dir):
#             for img in os.listdir(test_dir):
#                 test_images.append(os.path.join(test_dir, img))
#         else:
#             print(f"Directory {test_dir} does not exist.")
#
#     return train_images, val_images, test_images

import torch
import os
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageFilter


def get_images(root_dir):
    images = []
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(dir, file))

    return images


class FolderDataSet(data.Dataset):

    def __init__(self, root_dir, input_transform=None, target_transform=None):
        super(FolderDataSet, self).__init__()
        self.files = get_images(root_dir)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        pred = Image.open(self.files[index]).convert('RGB')
        target = pred.copy()

        if self.input_transform:
            pred = pred.filter(ImageFilter.GaussianBlur(2))
            pred = self.input_transform(pred)

        if self.target_transform:
            target = self.target_transform(target)

        return pred, target

    def __len__(self):
        return len(self.files)


def get_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def get_train_transforms(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


def get_target_transforms(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])


def get_train_dataset(root, crop_size, upscale_factor):
    crop_size = get_crop_size(crop_size, upscale_factor)
    return FolderDataSet(
        root_dir=root,
        input_transform=get_train_transforms(crop_size),
        target_transform=get_target_transforms(crop_size)
    )

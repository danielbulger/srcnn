import os
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageFilter


def get_images(root_dir):
    """
    Gets all the images from a directory recursively.
    :param root_dir: The starting directory to search for images.
    :return: A list of image files.
    """
    images = []
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(dir, file))

    return images


class FolderDataSet(data.Dataset):

    def __init__(self, root_dir, crop_size):
        """
        Creates a new dataset of images from the provided root directory.
        :param root_dir: The folder to search for images.
        :param crop_size: The size to crop the images to.
        """
        super(FolderDataSet, self).__init__()
        self.files = get_images(root_dir)
        self.input_transform = get_train_transforms(crop_size)
        self.target_transform = get_train_transforms(crop_size)

    def __getitem__(self, index):
        pred = Image.open(self.files[index]).convert('RGB')
        target = pred.copy()

        pred = pred.filter(ImageFilter.GaussianBlur(2))
        pred = self.input_transform(pred)

        target = self.target_transform(target)

        return pred, target

    def __len__(self):
        return len(self.files)


def get_crop_size(crop_size, upscale_factor):
    """
    Calculates the size that images should be crop to.
    This keeps images in the required aspect ratio regardless of the upscale value.
    :param crop_size: The base cropping size.
    :param upscale_factor: The amount the images are upscaled by.
    :return:
    """
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
    """
    Get the training dataset.
    :param root: The folder to search for images in.
    :param crop_size: The base cropping size.
    :param upscale_factor: The amount the target images are upscaled by.
    :return:
    """
    crop_size = get_crop_size(crop_size, upscale_factor)
    return FolderDataSet(
        root_dir=root,
        crop_size=crop_size
    )

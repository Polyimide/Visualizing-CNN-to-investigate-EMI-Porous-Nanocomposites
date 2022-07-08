from torch.utils.data import Dataset
from skimage import io
import os
from PIL import Image


class PorousData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        # return the size of whole dataset
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = io.imread(img_path)
        img = Image.fromarray(img)

        label = img_path.split("\\")[-1].split("_")[0]
        # sample = {'image': img, 'label': label}
        if label == "R":
            label_number = 0
        elif label == 'A':
            label_number = 1
        else:
            raise AttributeError

        if self.transform:
            img = self.transform(img)

        sample = [img, label_number]
        return sample
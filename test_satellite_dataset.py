import os
from PIL import Image
from torch.utils.data import Dataset

class TestSatelliteDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Initialize class for test dataset
        
        - image_dir: directory containing the input images.
        - transform: optional transformation
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # list all files in image_dir that end with .jpg or .png
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name  # returns the image and its file name
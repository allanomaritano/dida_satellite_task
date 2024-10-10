import os  
from PIL import Image  # Image class from PIL for image manipulation
from torch.utils.data import Dataset  # Importing Dataset class from PyTorch for custom datasets

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Initialize the SatelliteDataset.
        
        - image_dir: directory containing the input images.
        - label_dir: directory containing the corresponding labels.
        - transform: optional transformation
        """
        # store the directorys
        self.image_dir = image_dir  
        self.label_dir = label_dir  
        self.transform = transform  
        
        # list all files in image_dir that end with .jpg or .png
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        #print(f"found {len(self.images)} images in {image_dir}") 

    def __len__(self):
        return len(self.images)  # length of the dataset =number of images

    def __getitem__(self, idx):
        """

        - idx: Index of the image/label pair to retrieve.

        Returns:
        - Tuple of (image, label) after transformations.
        """
        img_name = self.images[idx]  # Get the image name at the specified index
        img_path = os.path.join(self.image_dir, img_name)  #path to the image
        
        # define potential naming conventions for the label files
        possible_label_names = [
            img_name.replace('.jpg', '_label.png').replace('.png', '_label.png'),  # e.g., img.jpg -> img_label.png
            img_name.replace('.jpg', '.png').replace('.png', '.png'),  # e.g., img.jpg -> img.png
            img_name  # Use the same name as the image for labels
        ]

        label_path = None  # Initialize label_path to None
        # Check each possible label file name to see if it exists
        for label_name in possible_label_names:
            temp_path = os.path.join(self.label_dir, label_name)  # path to label
            if os.path.exists(temp_path):  # Check if the label file exists
                label_path = temp_path  # If found, assign it to label_path
                break  # Exit loop once a label is found

        if label_path is None:  # if no label file was found
            print(f"Warning: No label found for image {img_name}")  # Print a warning
            # Create a blank label image of the same size as the input image
            image = Image.open(img_path).convert("RGB")  # Open the image and convert it to RGB format
            label = Image.new("L", image.size, 0)  # Create a blank label (black) with the same dimensions as the image
        else:  # if a label file was found
            image = Image.open(img_path).convert("RGB")  # open and convert the image to RGB
            label = Image.open(label_path).convert("L")  # convert label to grayscale

        # Apply transformations 
        if self.transform:
            image = self.transform(image)  
            label = self.transform(label)  
        
        return image, label  # Return image with label

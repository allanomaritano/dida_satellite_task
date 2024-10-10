import torch
from test_satellite_dataset import TestSatelliteDataset
from unet import UNet  # Import your UNet model
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def load_model(model_path):
    """
    Load trained UNet model from file.

    -Args: model_path (str): Path to the saved model file.

    Returns loaded and initialized UNet model in evaluation mode.
    """
    model = UNet()  # Initialize an instance of the UNet model
    model.load_state_dict(torch.load(model_path))  # Load the saved model weights
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    return model

def predict(model, dataloader, output_dir):
    """
    Generate predictions using model and save the results.

    Args:
        model (UNet): The loaded UNet model.
        dataloader (DataLoader) containing the test images.
        output_dir (str): Directory to save the prediction results.
    """
    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the appropriate device

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            
            # Assuming binary segmentation, apply threshold
            predictions = (outputs > 0.5).float()

            # Save predictions with the same name as the input image
            image_filenames = dataloader.dataset.images  # Access original image filenames
            for j, pred in enumerate(predictions):
                pred_img = transforms.ToPILImage()(pred.squeeze().cpu())
                
                # Get the filename corresponding to this image
                original_filename = image_filenames[i * dataloader.batch_size + j]
                # output path with the same image name
                pred_img.save(os.path.join(output_dir, original_filename))

def main2():
    """
    Main (main2) function to orchestrate the prediction process.
    """
    # Define file paths
    model_path = 'models/unet_model.pth'  
    test_dir = 'data/test/images'         
    output_dir = 'data/test/masks'       

    #trained model
    model = load_model(model_path)

    # test dataset with transformations: resize images to 256x256 and convert to PyTorch tensors
    test_dataset = TestSatelliteDataset(test_dir, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]))

    # Create a DataLoader for efficient batching and loading of test images
    # batch_size=1 ensures we process one image at a time
    # shuffle=False maintains the order of test images
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Generate predictions using the loaded model and save the results
    predict(model, test_loader, output_dir)

    # print that process is complete
    print("Prediction completed. Results saved in", output_dir)

if __name__ == '__main__':
    main2()
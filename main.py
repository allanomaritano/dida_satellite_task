import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet
from satellite_dataset import SatelliteDataset

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0  # initialize total loss
    for images, labels in train_loader:  # Loop through batches of data
        images, labels = images.to(device), labels.to(device)  #data to device(cpu)
        optimizer.zero_grad()  # Reset gradients to zero before backpropagation
        outputs = model(images)  # Forward pass: Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass: Compute gradients
        optimizer.step()  # Update model weights using optimizer
        total_loss += loss.item()  # Accumulate loss for this batch
    
    return total_loss / len(train_loader)  # average loss over the entire training set



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #gpu/cpu
    print(f"Using device: {device}")

    # data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    #datasets, directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'data', 'train', 'images')
    label_dir = os.path.join(base_dir, 'data', 'train', 'labels')
    #check
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Error: Image directory ({image_dir}) or label directory ({label_dir}) does not exist.")
        return
    # create dataset and loader
    train_dataset = SatelliteDataset(image_dir, label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Initialize the model, loss function, and optimizer
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train loop
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Save model
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'unet_model.pth'))
    print("Training completed. Model saved.")

if __name__ == '__main__':
    main()




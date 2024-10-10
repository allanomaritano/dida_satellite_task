import torch  # PyTorch for deep learning
import torch.nn as nn  # for neural network layers and functions

# Class for a double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        The __init__ method initializes the layers of the DoubleConv block.
        It contains two convolutional layers followed by batch normalization and ReLU activation.
        
        Parameters:
        in_channels: Number of input channels  (3 for RGB images)
        out_channels: Number of output channels (or filters) produced by the convolution
        """
        # parent class (nn.Module) to get its methods and properties
        super(DoubleConv, self).__init__()

        # nn.Sequential groups a series of layers that will be executed sequentially
        self.conv = nn.Sequential(
            # First 2D convolution layer:
            # in_channels: Number of input channels (e.g., 3 for RGB images)
            # out_channels: Number of filters or feature maps (increases feature abstraction)
            # kernel_size=3: 3x3 convolution filter
            # stride=1: moves the filter by 1 pixel at a time
            # padding=1: 1-pixel around the input for spatial dimensions
            # bias=False: not needed due to batch normalization
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),

            #batch normalization for standard the output from the convolution layer, helps stabilize and speed up training
            nn.BatchNorm2d(out_channels),

            # ReLU (Rectified Linear Unit) introduces non-linearity. 
            # replaces negative values in the feature map with 0
            # inplace=True saves memory in the input itsself
            nn.ReLU(inplace=True),

            # Second 2D convolution layer:
            # out_channels for both input and output channels,
            # -> this convolution layer processes the same feature maps produced by the first layer.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),

            # Batch normalization and ReLU after the second convolution layer for stable and faster learning.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        The forward method defines how the input tensor 'x' passes through the layers in this block.
       
        x is the image (parameter).
        returns the tensor after convolutions
        """
        return self.conv(x)  # Pass the input through the layers defined in nn.Sequential

#U-Net model class
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, features=[64, 128, 256, 512]):
        """
        U-Net architecture with specified parameters.
        
        - features (list): List defining the number of output channels for each DoubleConv block in the network.
        """
        #parent class 
        super(UNet, self).__init__()

        #layers for the downsampling and upsampling paths of the U-Net
        self.ups = nn.ModuleList()  # list for upsampling layers
        self.downs = nn.ModuleList()  # list for downsampling layers
        
        # Max pooling reduces the spatial dimensions of the feature maps by two
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling 
        for feature in features:
            # Append a DoubleConv block to the downs list to learn features at each level
            self.downs.append(DoubleConv(n_channels, feature))
            n_channels = feature  # Update the number of channels for the next layer

        # Upsampling
        for feature in reversed(features):
            # Create a transposed convolution layer for upsampling
            # The output channels are double the current feature size (due to concatenation)
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            # Append a DoubleConv block to refine the features after upsampling
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck layer, which is the deepest part of the U-Net, increasing feature channels
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # final convolution layer that reduces the feature maps to the desired number of output classes
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    # forward pass for the U-Net model
    def forward(self, x):
        """
        Defines the forward pass of the network.

        - x (tensor): Input tensor with shape (N, C, H, W), where N is batch size, 
                      C is number of channels, H is height, and W is width.

        returns tensor containing the segmentation map.
        """
        skip_connections = []  # list to store feature maps for skip connections

        # Downsampling (contracting path)
        for down in self.downs:
            x = down(x)  # Pass the input through the DoubleConv block to extract features
            skip_connections.append(x)  # Save the output for the skip connection
            x = self.pool(x)  # Apply max pooling to reduce the spatial dimensions by half

        # Pass through the bottleneck layer for additional feature learning
        x = self.bottleneck(x)  

        # Reverse the order of skip connections for the upsampling path
        skip_connections = skip_connections[::-1]  # Prepare for concatenation in the upsampling phase

        # Upsampling (expansive path)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Apply upsampling to increase spatial dimensions
            
            # skip connection for concatenation
            skip_connection = skip_connections[idx // 2]  

            # ensure that the dimensions of the upsampled feature map match those of the skip connection
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])  # Adjust size using interpolation

            # Concatenate the skip connection with the upsampled output along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Pass the concatenated features through the DoubleConv block to refine the output
            x = self.ups[idx + 1](concat_skip)

        # final 1x1 convolution to get the output segmentation map with class scores
        return self.final_conv(x)
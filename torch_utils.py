import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# load model
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels    # number of channels in the input
        self.n_classes = n_classes      # number of output classes

        # Define the U-Net architecture using DoubleConv modules
        self.inc = DoubleConv(n_channels, 64)   # initial double convolution
        self.down1 = DoubleConv(64, 128)        # downscale path double convolution
        self.down2 = DoubleConv(128, 256)       # further downscale path double convolution
        self.up1 = DoubleConv(256 + 128, 128)   # upscale path double convolution
        self.up2 = DoubleConv(128 + 64, 64)     # further upscale path double convolution
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) # final 1x1 convolution to map to class scores

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool2d(x1, 2))
        x3 = self.down2(F.max_pool2d(x2, 2))

        # Decoder path
        x = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x)

        # Final convolution to get class logits
        logits = self.outc(x)
        return logits

# Define the number of channels in the input images and the number of classes for binary segmentation
n_channels = 3  # Number of channels in the input images (RGB)
n_classes = 1   # Number of classes (1 for binary segmentation)

# Create an instance of the UNet model with specified channel and class count
model = UNet(n_channels=n_channels, n_classes=n_classes)

PATH = os.path.join(os.path.dirname(__file__), 'model_state_dict_model_after_testing1.pth')
model.load_state_dict(torch.load(PATH, map_location=device))

model.eval()

def process_lane(filename):
    # Load an image
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  
    image = image.transpose((2, 0, 1))  
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def model_inference(image_tensor):
    with torch.no_grad():  # No need to compute gradients
        output = model(image_tensor)
        output = torch.sigmoid(output)  # Assuming a sigmoid activation at the output
        output = output.squeeze(0).cpu().numpy()  # Remove batch dimension and move to cpu
    return output

def create_output_image(original_image, prediction):
    lane_mask = (prediction[0] > 0.2).astype(np.uint8)  # Threshold predictions to create a binary mask
    lane_mask = cv2.resize(lane_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(original_image)
    colored_mask[lane_mask == 1] = [0, 0, 225]  # Red mask
    output_image = cv2.addWeighted(original_image, 1, colored_mask, 0.4, 0)
    return output_image


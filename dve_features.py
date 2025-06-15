import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms

class DVEFeatureExtractor(nn.Module):
    """
    Standalone DVE Feature Extractor for animal re-identification
    Based on the "Addressing the Elephant in the Room" paper
    """
    def __init__(self, backbone='resnet50', dve_dim=64, pretrained=True):
        super(DVEFeatureExtractor, self).__init__()
        self.dve_dim = dve_dim
        
        # Use SE-ResNet50 backbone (first 3 layers for DVE features)
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Extract first 3 convolutional blocks
            self.feature_extractor = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,  # First ResNet block
                self.backbone.layer2,  # Second ResNet block  
                self.backbone.layer3   # Third ResNet block (factor 4 downscaling)
            )
            input_channels = 1024  # Output channels from layer3
        
        # DVE convolution layer applied after 3rd layer
        self.dve_conv = nn.Conv2d(input_channels, dve_dim, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(dve_dim)
        
    def forward(self, x):
        """
        Extract DVE features from input image
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            dve_features: Part-aware descriptors (B, dve_dim, H/4, W/4)
        """
        # Extract features through first 3 backbone layers
        features = self.feature_extractor(x)
        
        # Apply DVE convolution and normalization
        dve_features = self.dve_conv(features)
        dve_features = self.bn(dve_features)
        dve_features = F.normalize(dve_features, p=2, dim=1)
        
        return dve_features

def compute_dve_loss(phi_x, phi_x_prime, phi_x_alpha, temperature=0.1):
    """
    Compute DVE loss for unsupervised part alignment
    """
    B, D, H, W = phi_x.shape
    
    # Flatten spatial dimensions
    phi_x_flat = phi_x.view(B, D, -1).permute(0, 2, 1)  # (B, HW, D)
    phi_x_prime_flat = phi_x_prime.view(B, D, -1).permute(0, 2, 1)
    phi_x_alpha_flat = phi_x_alpha.view(B, D, -1).permute(0, 2, 1)
    
    # Compute similarity matrices
    sim_xx_prime = torch.bmm(phi_x_flat, phi_x_prime_flat.permute(0, 2, 1))
    sim_x_alpha = torch.bmm(phi_x_flat, phi_x_alpha_flat.permute(0, 2, 1))
    
    # Apply temperature scaling
    sim_xx_prime = sim_xx_prime / temperature
    sim_x_alpha = sim_x_alpha / temperature
    
    # Compute probability distributions
    prob_xx_prime = F.softmax(sim_xx_prime, dim=-1)
    prob_x_alpha = F.softmax(sim_x_alpha, dim=-1)
    
    # DVE loss: KL divergence between correspondence probabilities
    dve_loss = F.kl_div(prob_xx_prime.log(), prob_x_alpha, reduction='batchmean')
    
    return dve_loss

def extract_dve_features(image_paths, output_dir, dve_dim=64, batch_size=8):
    """
    Extract DVE features from a list of image paths
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize DVE model
    model = DVEFeatureExtractor(dve_dim=dve_dim).to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    h5_path = Path(output_dir) / "descriptors.h5"
    
    with h5py.File(h5_path, "w") as h5:
        for img_path in tqdm(image_paths, desc="Extracting DVE features"):
            img_id = Path(img_path).stem
            
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARN] Cannot read {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dve_features = model(image_tensor)
                
            # Flatten spatial dimensions and convert to numpy
            dve_flat = dve_features.squeeze(0).view(dve_dim, -1).permute(1, 0)
            dve_np = dve_flat.cpu().numpy().astype(np.float32)
            
            h5.create_dataset(img_id, data=dve_np, compression="gzip")
    
    print(f"DVE features saved to {h5_path}")
import torch
import torch.nn as nn

class SimpleSAR_CNN(nn.Module):
    def __init__(self):
        super(SimpleSAR_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        feature_map = self.encoder(x)
        clean_out = self.decoder(feature_map)
        return clean_out + x
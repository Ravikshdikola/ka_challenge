import torch
import torch.nn as nn

class BalancedDeepModel(nn.Module):
    def __init__(self, output_dim=10):
        super(BalancedDeepModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),   # Small dropout after activation

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(128, output_dim)  # Final layer, no BN or Dropout here
        )

    def forward(self, x):
        return self.model(x)
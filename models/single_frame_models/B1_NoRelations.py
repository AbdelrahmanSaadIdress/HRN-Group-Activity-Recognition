import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class b1_norelations_stage1(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape: (N, 2048, 1, 1)

        # New classifier
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """
        x: (B, P, 3, 224, 224)
        """
        B, P, C, H, W = x.shape

        # (B*P, 3, 224, 224)
        x = x.view(B * P, C, H, W)

        # (B*P, 2048, 1, 1)
        x = self.backbone(x)

        # (B*P, 2048)
        x = torch.flatten(x, 1)

        # (B*P, 9)
        x = self.fc(x)

        return x


class b1_norelations_stage2(nn.Module):
    def __init__(self, stage1_model, num_classes=8):
        super().__init__()

        # Initialize the backbone from stage1_model
        self.stage1 = stage1_model.backbone

        self.dense_layer = nn.Linear(2048, 128)

        # Group activity classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes), 
        )

    def forward(self, x):
        """
        x: (B, P, 3, 224, 224)
        """

        B, P, C, H, W = x.shape

        # (B*P, 3, 224, 224)
        x = x.view(B * P, C, H, W)

        # (B*P, 2048, 1, 1)
        x = self.stage1(x)

        # (B*P, 2048)
        x = torch.flatten(x, 1)

        # (B, P, 2048)
        x = x.view(B, P, 2048)

        # (B, P, 128)
        x = self.dense_layer(x)

        # Average pooling over players
        team_one = x[:,:6,:].mean(dim=1)  # (B, 128)
        team_two = x[:,6:,:].mean(dim=1)  # (B, 128)

        # (B, 2, 128)
        x = torch.concat([team_one, team_two], dim=1)

        # (B, 256)
        x = x.view(B, -1)

        # (B, 8)
        x = self.fc(x)

        return x
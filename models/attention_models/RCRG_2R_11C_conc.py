import torch
import torch.nn as nn

import itertools

from ..single_frame_models import b1_norelations_stage1
from .RelationalUnit import RelationLayer

class RCRG_2R_11C_conc_stage2(nn.Module):
    def __init__(self, 
                stage1_model, in_channels=2048,
                hidden_channels=512, output_channels=[256, 128],
                num_classes=8
                ):
        super().__init__()

        # Initialize the backbone from stage1_model
        self.stage1 = stage1_model.backbone

        for param in self.stage1.parameters():
            param.requires_grad = False 
        
        self.relational_unit_one = RelationLayer(in_channels, hidden_size=hidden_channels, out_channels=output_channels[0])
        self.relational_unit_two = RelationLayer(in_channels, hidden_size=hidden_channels, out_channels=output_channels[1])
        
        self.lstm1 = nn.LSTM(
            input_size=2048,
            hidden_size=in_channels,
            batch_first=True
        ) 

        self.lstm2 = nn.LSTM(
            input_size=sum(output_channels),
            hidden_size=sum(output_channels),
            batch_first=True
        ) 

        # Example Stage 2 classifier
        self.classifier = nn.Sequential(
            nn.Linear(12*sum(output_channels), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )

    def forward(self, x):
        """
        x: (B, seq, P, 3, 224, 224)
        """

        B, S, P, C, H, W = x.shape

        # (B*S*P, 3, 224, 224)
        x = x.view(B * S * P, C, H, W)

        # Feature extraction
        with torch.no_grad():  # optional but saves memory
            x = self.stage1(x)      # (B*S*P, 2048, 1, 1)

        # (B*S*P, 2048)
        x = torch.flatten(x, 1)
        # (B*P, S, 2048)
        x = x.view(B*P, S, -1 )
        # (B*P, S, 1024)
        x, (h, c) = self.lstm1(x)   
        # (B*S, p, 1024)
        x = x.contiguous().view(B*S, P, -1 )

        # (2, 132)
        edge_index = torch.tensor([(i, j) for i, j in itertools.permutations(range(P), 2)]).t().to(x.device)
        # (B*S, P, output_channels_i)
        x_1 = self.relational_unit_one(x=x, edge_index=edge_index)
        x_2 = self.relational_unit_two(x=x, edge_index=edge_index)
        # (B*S, P, (output_channels_1+output_channels_2))
        x = torch.cat([x_1, x_2], dim=2)

        # (B*P, S, (output_channels_1+output_channels_2))
        x = x.contiguous().view(B*P, S, -1)
        x, (h, c) = self.lstm2(x) 

        # (B*P, (output_channels_1+output_channels_2))
        x = x[:,-1,:]

        # (B, P * (output_channels_1+output_channels_2) )
        x = x.view(B, -1)
        # (B, 8)
        x = self.classifier(x)

        return x

# # Create Stage 1 model
# stage1_model = b1_norelations_stage1(num_classes=9)
# # Create Stage 2 model
# model = RCRG_2R_11C_conc_stage2(stage1_model)
# # Random input
# B = 2
# P = 12
# S = 9
# x = torch.randn(B, S, P, 3, 224, 224)

# # Forward pass
# with torch.no_grad():
#     out = model(x)

# print("Input shape :", x.shape)
# print("Output shape:", out.shape)




# total = sum(p.numel() for p in model.parameters())
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total params: {total:,}")
# print(f"Trainable params: {trainable:,}")




# # from torchinfo import summary
# # summary(
# #     model,
# #     input_size=(4, 12, 3, 224, 224),  # (B, P, C, H, W)
# #     device="cpu"
# # )


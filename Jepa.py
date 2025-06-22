import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class JEPA(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, lr=1e-3): 
        super().__init__()
        self.encoder_context = Encoder(input_dim, hidden_dim)
        self.encoder_target = Encoder(input_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # ✅ Add optimizer

    def forward(self, context, target):
        z_context = self.encoder_context(context)
        z_target = self.encoder_target(target)
        z_pred = self.predictor(z_context)
        return z_pred, z_target

    def compute_loss(self, context, target):
        z_pred, z_target = self.forward(context, target)
        return F.mse_loss(z_pred, z_target)

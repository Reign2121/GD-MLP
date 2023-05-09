import torch
import torch.nn as nn
import torch.nn.functional as F

class gated_mlp (nn.Module):

    """
    MLP block
    """
    def __init__(self, num_features, pred_len, hidden_units):
        super(gated_mlp, self).__init__()
        self.num_features = num_features
        self.pred_len = pred_len
        self.hidden_units = hidden_units
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        self.trend_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        )


        self.residual_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.Sigmoid()) #gate

        self.layer = nn.Linear(self.hidden_units, self.hidden_units)


    def forward(self, x):
        
        residual_train,trend_train = self.decomposition(x)

        # trend MLP (gated)
        trend_mlp = self.trend_mlp(trend_train)
        gate_t = self.gate(trend_mlp)
        trend_mlp = trend_mlp * gate_t
        trend_mlp = self.layer(trend_mlp)
        

        # residual MLP(gated)
        residual_mlp = self.residual_mlp(residual_train)
        gate_r = self.gate(residual_mlp)
        residual_mlp = residual_mlp * gate_r
        residual_mlp = self.layer(residual_mlp)

        return trend_mlp, residual_mlp

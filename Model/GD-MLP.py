import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        res = x - trend
        return res, trend



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



class gated_sum (nn.Module):

    """
    Composing block with gate
    """
     
    def __init__(self, num_features, pred_len, hidden_units):
        super(gated_sum, self).__init__()
        self.num_features = num_features
        self.pred_len = pred_len
        self.hidden_units = hidden_units
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.gated_mlp = gated_mlp(num_features, pred_len, hidden_units) # 부모 클래스의 상속받은 인스턴스 다시 쓰려면 초기화해줘야 한다.

        self.trend_weight = nn.Sequential(
            nn.Linear(self.hidden_units, 1), #가중합 하기 위한 gate
            nn.Sigmoid()
        )

        self.residual_weight = nn.Sequential(
            nn.Linear(self.hidden_units, 1),  #가중합 하기 위한 gate
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Linear(self.hidden_units,self.pred_len)

    def forward(self, x):
        trend_mlp, residual_mlp = self.gated_mlp(x)  
        # combine trend and residual MLPs with weighted sum
        trend_weight = self.trend_weight(trend_mlp)
        residual_weight = self.residual_weight(residual_mlp)

        weighted_sum = trend_mlp * trend_weight + residual_mlp * residual_weight

        # output layer
        outputs = self.output_layer(weighted_sum)

        return outputs

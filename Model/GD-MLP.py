import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    * decomposition Source from autoformer, Dlinear
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
    * decomposition Source from autoformer, Dlinear
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
    MLP block(gated_mlp)
    """
    def __init__(self, num_features, pred_len, hidden_units):
        super(gated_mlp, self).__init__()
        self.num_features = num_features #피처 수
        self.pred_len = pred_len # 예측 길이
        self.hidden_units = hidden_units #노드 수
        kernel_size = 25 #same as Autoformer, Dlinear
        self.decomposition = series_decomp(kernel_size)

        self.trend_mlp = nn.Sequential( 
            nn.Linear(self.num_features, self.hidden_units), 
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        ) # MLP for Trend


        self.residual_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        ) # MLP for Residual

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.Sigmoid()) #gate for both Trend and Residual

        self.layer = nn.Linear(self.hidden_units, self.hidden_units)


    def forward(self, x):
        
        residual_train,trend_train = self.decomposition(x)

        # trend MLP (gated)
        trend_mlp = self.trend_mlp(trend_train) #MLP 통과
        gate_t = self.gate(trend_mlp) #gate 통과
        trend_mlp = trend_mlp * gate_t #element-wise product #torch.mul(residual_mlp, gate_r)
        trend_mlp = self.layer(trend_mlp)
        

        # residual MLP(gated)
        residual_mlp = self.residual_mlp(residual_train) #MLP 통과
        gate_r = self.gate(residual_mlp) #gate 통과
        residual_mlp = residual_mlp * gate_r #element-wise product #torch.mul(residual_mlp, gate_r)
        residual_mlp = self.layer(residual_mlp)

        return trend_mlp, residual_mlp



class gated_sum (nn.Module):

    """
    Composing block with gate
    """
     
    def __init__(self, num_features, pred_len, hidden_units):
        super(gated_sum, self).__init__()
        self.num_features = num_features #same as gated mlp
        self.pred_len = pred_len
        self.hidden_units = hidden_units
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.gated_mlp = gated_mlp(num_features, pred_len, hidden_units) # 상속받은 부모 클래스의 인스턴스의 속성을 다시 쓰기 위해 초기화해줘야 한다.
        
        self.output_layer = nn.Linear(self.hidden_units,self.pred_len) #pred_len으로 산출하기 위한 output_layer
        
        self.trend_weight = nn.Sequential(
            nn.Linear(self.hidden_units, 1), #가중합 하기 위한 gate  #1은 A/B testing(A:1, B:pred_len) 예정
            nn.Sigmoid() 
        )

        self.residual_weight = nn.Sequential(
            nn.Linear(self.hidden_units, 1),  #가중합 하기 위한 gate  #1은 A/B testing(A:1, B:pred_len) 예정
            nn.Sigmoid()
        )
        

    def forward(self, x):
        
        # gated_mlp
        trend_mlp, residual_mlp = self.gated_mlp(x)  #gated_mlp 블록을 통과
        
        # output layer
        output_trend = self.output_layer(trend_mlp)  #trend_output
        output_residual = self.output_layer(residual_mlp) #residual_output
        
        # combine trend and residual MLPs with weighted sum
        trend_weight = self.trend_weight(output_trend) # gate 통과
        residual_weight = self.residual_weight(output_residual) # gate 통과

        weighted_sum = (trend_mlp * trend_weight) + (residual_mlp * residual_weight) # Weighted sum == Final Output

        return weighted_sum #Final Output

import torch
import torch.nn as nn
import torch.nn.functional as F

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

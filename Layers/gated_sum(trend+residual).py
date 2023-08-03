import torch
import torch.nn as nn
import torch.nn.functional as F

class gated_sum (nn.Module):

    """
    Composing block with gate
    """
     
    def __init__(self, seq_len, num_features, pred_len, hidden_units):
        super(gated_sum, self).__init__()
        self.seq_len = seq_len #인풋 길이
        self.num_features = num_features #same as gated mlp
        self.pred_len = pred_len
        self.hidden_units = hidden_units
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.gated_mlp = gated_mlp(seq_len, num_features, pred_len, hidden_units) 
        
        self.output_layer = nn.Linear(self.hidden_units,self.pred_len) #pred_len으로 산출하기 위한 output_layer
        
        #Output gate
        self.gated_trend = nn.Sequential(
            nn.Linear(self.pred_len, 1), #가중합 하기 위한 gate  #1은 A/B testing(A:1, B:pred_len) 예정
            nn.Sigmoid() 
        )

        #self.gated_residual = nn.Sequential(
        #    nn.Linear(self.pred_len, 1),  #가중합 하기 위한 gate  #1은 A/B testing(A:1, B:pred_len) 예정
        #    nn.Sigmoid())
        

    def forward(self, x):
        
        # gated_mlp
        trend_mlp, residual_mlp = self.gated_mlp(x)  #gated_mlp 블록을 통과 #->torch.Size([8, 1, 512])

        # output layer
        output_trend = self.output_layer(trend_mlp)  #output_layer 통과 #->torch.Size([8, 1, 96])
        output_residual = self.output_layer(residual_mlp) #output_layer 통과 #->torch.Size([8, 1, 96])
        
        # combine trend and residual MLPs with weighted sum
        trend_weight = self.gated_trend(output_trend) # gate 통과 #->torch.Size([8, 1, 1])
        residual_weight = (1 - trend_weight)

        #trend_weight,residual_weight = trend_weight.permute(0,2,1), residual_weight.permute(0,2,1)

        weighted_sum = (output_trend * trend_weight) + (output_residual * residual_weight) # Weighted sum == Final Output #->torch.Size([8, 1, 96])
        
        return weighted_sum.permute(0,2,1) #Final Output #->torch.Size([8, 96, 1])

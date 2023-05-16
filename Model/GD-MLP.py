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
        x = torch.cat([front, x, end], dim=1) # -> torch.Size([8, 360, 8])
        x = self.avg(x.permute(0, 2, 1)) # -> torch.Size([8, 8, 336])
        x = x.permute(0, 2, 1) # -> torch.Size([8, 336, 8])
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
        return res, trend #torch.Size([8, 336, 8])


class gated_mlp (nn.Module):

    """
    MLP block(gated_mlp)
    """
    def __init__(self, seq_len, num_features, pred_len, hidden_units):
        super(gated_mlp, self).__init__()
        self.seq_len = seq_len #인풋 길이
        self.num_features = num_features #피처 수
        self.pred_len = pred_len # 예측 길이
        self.hidden_units = hidden_units #노드 수
        kernel_size = 25 #same as Autoformer, Dlinear
        self.decomposition = series_decomp(kernel_size)
        
        self.input_layer = nn.Linear(self.num_features,1) #차원 축소

        self.input_gate_t = nn.Sequential(
            nn.Linear(self.seq_len, 1),
            nn.Sigmoid()) #Input_gate for Trend

        self.input_gate_r = nn.Sequential(
            nn.Linear(self.seq_len, 1),
            nn.Sigmoid()) #Input_gate for Residual    

        self.trend_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_units), 
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        ) # MLP for Trend


        self.residual_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        ) # MLP for Residual



    def forward(self, x):
        
        residual_train,trend_train = self.decomposition(x) # torch.Size([8, 336, 8])

        #input_layer 
        trend_train = self.input_layer(trend_train) #-> torch.Size([8, 336, 1])
        residual_train = self.input_layer(residual_train)

        residual_train,trend_train = residual_train.permute(0,2,1), trend_train.permute(0,2,1) #-> torch.Size([8, 1, 336])

        #input_gate
        i_gate_t = self.input_gate_t(trend_train) #-> torch.Size([8, 1, 336])
        trend_train = trend_train * i_gate_t #-> torch.Size([8, 1, 336])

        i_gate_r = self.input_gate_r(residual_train) #-> torch.Size([8, 1, 336])
        residual_train = residual_train * i_gate_r #-> torch.Size([8, 1, 336])


        # trend MLP (gated)
        trend_mlp = self.trend_mlp(trend_train) #MLP 통과 #-> torch.Size([8, 1, 512])

        # residual MLP(gated)
        residual_mlp = self.residual_mlp(residual_train) #MLP 통과 #-> torch.Size([8, 1, 512])
      

        return trend_mlp, residual_mlp #torch.Size([8, 1, 512])



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
        
        #Variation gate
        self.gated_trend = nn.Sequential(
            nn.Linear(self.pred_len, 1), #가중합 하기 위한 gate  #1은 A/B testing(A:1, B:pred_len) 예정
            nn.Sigmoid() 
        )

        self.gated_residual = nn.Sequential(
            nn.Linear(self.pred_len, 1),  #가중합 하기 위한 gate  #1은 A/B testing(A:1, B:pred_len) 예정
            nn.Sigmoid()
        )
        

    def forward(self, x):
        
        # gated_mlp
        trend_mlp, residual_mlp = self.gated_mlp(x)  #gated_mlp 블록을 통과 #->torch.Size([8, 1, 512])

        # output layer
        output_trend = self.output_layer(trend_mlp)  #output_layer 통과 #->torch.Size([8, 1, 96])
        output_residual = self.output_layer(residual_mlp) #output_layer 통과 #->torch.Size([8, 1, 96])
        
        # combine trend and residual MLPs with weighted sum
        trend_weight = self.gated_trend(output_trend) # gate 통과 #->torch.Size([8, 1, 1])
        residual_weight = self.gated_residual(output_residual) # gate 통과 #->torch.Size([8, 1, 1])

        #trend_weight,residual_weight = trend_weight.permute(0,2,1), residual_weight.permute(0,2,1)

        weighted_sum = (output_trend * trend_weight) + (output_residual * residual_weight) # Weighted sum == Final Output #->torch.Size([8, 1, 96])
        
        return weighted_sum.permute(0,2,1) #Final Output #->torch.Size([8, 96, 1])

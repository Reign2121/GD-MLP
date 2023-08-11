import torch
import torch.nn as nn
import torch.nn.functional as F

class gated_mlp (nn.Module):

    """
    MLP block(gated_mlp)
    """
    def __init__(self, seq_len, num_features, pred_len, hidden_units):
        super(gated_mlp, self).__init__()
        self.seq_len = seq_len #인풋 길이 #ex)336
        self.num_features = num_features #피처 수
        self.pred_len = pred_len # 예측 길이 
        self.hidden_units = hidden_units #노드 수 #ex)512
        kernel_size = 25 #same as Autoformer, Dlinear
        self.decomposition = series_decomp(kernel_size)
        
        self.input_layer = nn.Linear(self.num_features,1)

        self.input_gate = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.Sigmoid()) #Input_gate


        self.trend_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_units),
            nn.Linear(self.hidden_units, self.hidden_units)
        ) # MLP for Trend


        self.residual_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_units),
            nn.Linear(self.hidden_units, self.hidden_units)
        ) # MLP for Residual



    def forward(self, x):
        
        residual_train,trend_train = self.decomposition(x) # torch.Size([8, 336, 1])

        #input_layer #not necessary
        trend_train = self.input_layer(trend_train) #-> torch.Size([8, 336, 1])
        residual_train = self.input_layer(residual_train)

        residual_train,trend_train = residual_train.permute(0,2,1), trend_train.permute(0,2,1) #-> torch.Size([8, 1, 336])
        
        #input_gate
        i_gate_t = self.input_gate(trend_train) #-> torch.Size([8, 1, 336])
        trend_train = trend_train * i_gate_t #-> torch.Size([8, 1, 336])

        i_gate_r = self.input_gate(residual_train) #-> torch.Size([8, 1, 336])
        residual_train = residual_train * i_gate_r #-> torch.Size([8, 1, 336])

        
        # trend MLP (gated)
        trend_mlp = self.trend_mlp(trend_train) #MLP 통과 #-> torch.Size([8, 1, 512])

        # residual MLP(gated)
        residual_mlp = self.residual_mlp(residual_train) #MLP 통과 #-> torch.Size([8, 1, 512])
      

        return trend_mlp, residual_mlp #torch.Size([8, 1, 512])

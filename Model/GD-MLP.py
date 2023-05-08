class GatedDecompositionMLP(nn.Module):
    def __init__(self, num_features, pred_len, hidden_units, moving_avg_kernel_size=25, moving_avg_stride=1):
        super(GatedDecompositionMLP, self).__init__()
        self.num_features = num_features
        self.pred_len = pred_len
        self.hidden_units = hidden_units
        self.moving_avg_kernel_size = moving_avg_kernel_size
        self.moving_avg_stride = moving_avg_stride

        self.trend_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        )

        self.gate_t = nn.Sigmoid(self.hidden_units, 1) #gate

        self.residual_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU()
        )

        self.gate_r = nn.Sigmoid(self.hidden_units, 1) #gate

        self.trend_weight = nn.Sequential(
            nn.Linear(self.hidden_units, 1),
            nn.Sigmoid()
        )

        self.residual_weight = nn.Sequential(
            nn.Linear(self.hidden_units, 1),
            nn.Sigmoid()
        )

        self.output_layer = nn.Linear(self.hidden_units, self.pred_len)

    def forward(self, input_trend, input_residual):
        # moving average to get the trend
        moving_average = nn.AvgPool1d(self.moving_avg_kernel_size, stride=self.moving_avg_stride, padding='same')

        # get the trend and residual of train data
        trend_train = moving_average(input_trend)
        residual_train = input_trend - trend_train

        # get the trend and residual of test data
        trend_test = moving_average(input_residual)
        residual_test = input_residual - trend_test

        # trend MLP (gated)
        trend_mlp = self.trend_mlp(trend_train)
        gate_t = self.gate_t(trend_mlp)
        h1 = nn.Linear(self.hidden_units, self.hidden_units)
        gate_t = h1.weight.data
        trend_mlp = h1(trend_mlp)

        # residual MLP(gated)
        residual_mlp = self.residual_mlp(residual_train)
        gate_r = self.gate_r(residual_mlp)
        h2 = nn.Linear(self.hidden_units, self.hidden_units)
        gate_r = h2.weight.data
        residual_mlp = h2(residual_mlp)
        
        # combine trend and residual MLPs with weighted sum
        trend_weight = self.trend_weight(trend_mlp)
        residual_weight = self.residual_weight(residual_mlp)

        weighted_sum = trend_mlp * trend_weight + residual_mlp * residual_weight

        # output layer
        outputs = self.output_layer(weighted_sum)

        return outputs

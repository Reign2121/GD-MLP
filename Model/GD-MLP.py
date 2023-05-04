import tensorflow as tf
#from statsmodels.tsa.seasonal import seasonal_decompose

class GatedDecompositionMLP:
    
    def __init__(self, num_features, num_outputs, hidden_units, moving_avg_kernel_size=25, moving_avg_stride=1):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_units = hidden_units
        self.moving_avg_kernel_size = moving_avg_kernel_size
        self.moving_avg_stride = moving_avg_stride
    
    def gated_residual_mlp(self, input):
        # feedforward network with gating and residual connection
        dense1 = tf.keras.layers.Dense(self.hidden_units, activation='relu')(input)
        gate1 = tf.keras.layers.Dense(self.hidden_units, activation='sigmoid')(input)
        gated_dense1 = tf.keras.layers.Multiply()([dense1, gate1])
        residual1 = tf.keras.layers.Add()([gated_dense1, input])

        # feedforward network with gating and residual connection
        dense2 = tf.keras.layers.Dense(self.hidden_units, activation='relu')(residual1)
        gate2 = tf.keras.layers.Dense(self.hidden_units, activation='sigmoid')(residual1)
        gated_dense2 = tf.keras.layers.Multiply()([dense2, gate2])
        residual2 = tf.keras.layers.Add()([gated_dense2, residual1])

        return residual2

    def gd_mlp(self, x_train, y_train, x_test, y_test):
        # moving average to get the trend
        moving_average = tf.keras.layers.AveragePooling1D(pool_size=self.moving_avg_kernel_size, strides=self.moving_avg_stride, padding= 'same')
        
        # get the trend and residual of train data
        trend_train = moving_average(x_train)
        residual_train = x_train - trend_train

        # get the trend and residual of test data
        trend_test = moving_average(x_test)
        residual_test = x_test - trend_test

        # input layers
        input_trend = tf.keras.layers.Input(shape=(None, self.num_features), name='input_trend')
       
        input_residual = tf.keras.layers.Input(shape=(None, self.num_features), name='input_residual')


        # trend MLP
        trend_mlp = self.gated_residual_mlp(input_trend)
        # residual MLP
        residual_mlp = self.gated_residual_mlp(input_residual)
        # combine trend and residual MLPs with weighted sum
        trend_weight = tf.keras.layers.Dense(1, activation='sigmoid', name='trend_weight')(trend_mlp)
        residual_weight = tf.keras.layers.Dense(1, activation='sigmoid', name='residual_weight')(residual_mlp)

        weighted_sum =  trend_mlp * trend_weight + residual_mlp * residual_weight
        print(weighted_sum)

        weighted_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(weighted_sum)

        # output layer
        outputs = tf.keras.layers.Dense(self.num_outputs, activation='linear')(weighted_sum)

        # define and compile model
        model = tf.keras.Model(inputs=[input_trend, input_residual],
                              outputs=outputs)
        model.compile(loss='mse', optimizer='adam')

        # train model
        model.fit([trend_train, residual_train],
                  y_train,
                  epochs=50,
                  batch_size=128,
                  validation_data=([trend_test, residual_test], y_test),
                  verbose=2)

        return model

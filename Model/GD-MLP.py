import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose

class GatedDecompositionMLP:
    
    def __init__(self, num_features, num_outputs, hidden_units):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_units = hidden_units
    
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


    def time_series_mlp(self, x_train, y_train, x_test, y_test):
        # seasonal decomposition
        result_train = seasonal_decompose(x_train[:, :, 0], model='additive', period=25)
        trend_train = result_train.trend
        seasonal_train = result_train.seasonal
        residual_train = result_train.resid

        result_test = seasonal_decompose(x_test[:, :, 0], model='additive', period=25)
        trend_test = result_test.trend
        seasonal_test = result_test.seasonal
        residual_test = result_test.resid

        # input layers
        input_trend = tf.keras.layers.Input(shape=(None, 1), name='input_trend')
        input_seasonal = tf.keras.layers.Input(shape=(None, 1), name='input_seasonal')
        input_residual = tf.keras.layers.Input(shape=(None, self.num_features-2), name='input_residual')

        # trend MLP
        trend_mlp = self.gated_residual_mlp(input_trend)

        # seasonal MLP
        seasonal_mlp = self.gated_residual_mlp(input_seasonal)

        # residual MLP
        residual_mlp = self.gated_residual_mlp(input_residual)

        # combine trend, seasonal, and residual MLPs with weighted sum
        trend_weight = tf.keras.layers.Dense(1, activation='sigmoid', name='trend_weight')(trend_mlp)
        seasonal_weight = tf.keras.layers.Dense(1, activation='sigmoid', name='seasonal_weight')(seasonal_mlp)
        residual_weight = tf.keras.layers.Dense(1, activation='sigmoid', name='residual_weight')(residual_mlp)

        weighted_sum = tf.keras.layers.Concatenate()([
            trend_mlp * trend_weight,
            seasonal_mlp * seasonal_weight,
            residual_mlp * residual_weight,
            ])
        weighted_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(weighted_sum)

        # output layer
        outputs = tf.keras.layers.Dense(self.num_outputs, activation='linear')(weighted_sum)

        # define and compile model
        model = tf.keras.Model(inputs=[input_trend, input_seasonal, input_residual],
                              outputs=outputs)
        model.compile(loss='mse', optimizer='adam')

        # train model
        model.fit([trend_train, seasonal_train, residual_train],
                  y_train,
                  epochs=50,
                  batch_size=128,
                  validation_data=([trend_test, seasonal_test, residual_test], y_test),
                  verbose=2)

        return model

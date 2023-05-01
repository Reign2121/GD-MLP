import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose

class GatedDecompositionMLP:
    
    def __init__(self, num_features, num_outputs, hidden_units):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_units = hidden_units
        
    def gated_mlp(self, inputs):
        mlp = tf.keras.layers.Dense(self.hidden_units, activation='relu')(inputs)
        mlp = tf.keras.layers.Dense(self.hidden_units, activation='relu')(mlp)
        gate = tf.keras.layers.Dense(self.hidden_units, activation='sigmoid')(inputs)
        gated_mlp = mlp * gate
        return gated_mlp
    
    def fit(self, x_train, y_train, x_test, y_test):
        # seasonal decomposition
        result_train = seasonal_decompose(x_train[:, :, 0], model='additive', period=24)
        trend_train = result_train.trend
        seasonal_train = result_train.seasonal
        residual_train = result_train.resid

        result_test = seasonal_decompose(x_test[:, :, 0], model='additive', period=24)
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
        self.model = tf.keras.Model(inputs=[input_trend, input_seasonal, input_residual],
                               outputs=outputs)
        self.model.compile(loss='mse', optimizer='adam')

        # train model
        self.model.fit([trend_train, seasonal_train, residual_train],
                  y_train,
                  epochs=50,
                  batch_size=128,
                  validation_data=([trend_test, seasonal_test, residual_test], y_test),
                  verbose=2)

    def predict(self, x):
        result = seasonal_decompose(x[:, :, 0], model='additive', period=24)
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        y_pred = self.model.predict([trend, seasonal, residual])
        return y_pred

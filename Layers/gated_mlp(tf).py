import tensorflow as tf

def gated_mlp(input, hidden_units):
    # feedforward network with gating and residual connection
    dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')(input)
    gate1 = tf.keras.layers.Dense(hidden_units, activation='sigmoid')(input)
    gated_dense1 = tf.keras.layers.Multiply()([dense1, gate1])
    residual1 = tf.keras.layers.Add()([gated_dense1, input]) #need to change shapes

    # feedforward network with gating and residual connection
    dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')(residual1)
    gate2 = tf.keras.layers.Dense(hidden_units, activation='sigmoid')(residual1)
    gated_dense2 = tf.keras.layers.Multiply()([dense2, gate2])
    residual2 = tf.keras.layers.Add()([gated_dense2, residual1])

    return residual2

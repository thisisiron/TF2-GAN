import tensorflow as tf  # TF 2.0


class Generator(tf.keras.Model):
    def __init__(self, num_classes, channels=1):
        super(Generator, self).__init__()
        self.channels = channels

        self.dense_z = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.combined_dense = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_x = tf.keras.layers.Dropout(0.5)

        self.final_dense = tf.keras.layers.Dense(28 * 28 * self.channels, activation='tanh')

        self.reshape = tf.keras.layers.Reshape((28, 28, self.channels))

    def call(self, inputs, labels, training=True):
        z = self.dense_z(inputs)
        # z = self.dropout_z(z, training)

        y = self.dense_y(labels)
        # y = self.dropout_y(y, training)

        combined_x = self.combined_dense(tf.concat([z, y], axis=-1))
        # combined_x = self.dropout_x(combined_x, training)

        x = self.final_dense(combined_x)

        return self.reshape(x)


class Discriminator(tf.keras.Model):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.flatten = tf.keras.layers.Flatten()

        self.maxout_z = MaxoutDense(240, k=5, activation='relu', drop_prob=0.5)
        self.maxout_y = MaxoutDense(50, k=5, activation='relu', drop_prob=0.5)
        self.maxout_x = MaxoutDense(240, k=4, activation='relu', drop_prob=0.5)

        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, labels, training=True):
        z = self.flatten(inputs)
        z = self.maxout_z(z, training)

        y = self.maxout_y(labels, training)

        x = self.maxout_x(tf.concat([z, y], axis=-1))
        
        return self.out(x)


class MaxoutDense(tf.keras.layers.Layer):
    def __init__(self, units, k, activation, drop_prob=0.5):
        self.dense = tf.keras.layers.Dense(units * k, activation=activation)
        self.dropout = tf.keras.layers.Dropout(drop_prob)
        self.reshape = tf.keras.layers.Reshape((-1, k, units))
        super(MaxoutDense, self).__init__()

    def call(self, inputs, training=True):
        x = self.dense(inputs)
        x = self.dropout(x, training)
        x = self.reshape(x)
        return tf.reduce_max(x, axis=1)


if __name__ == "__main__":
    pass

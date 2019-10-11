import tensorflow as tf  # TF 2.0


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(28 * 28 * 1, activation='tanh')

        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(inputs)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.reshape(x)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense_4(x)


if __name__ == "__main__":
    pass

import tensorflow as tf  # TF 2.0


class Generator(tf.keras.Model):
    def __init__(self, channels=1):
        super(Generator, self).__init__()
        self.channels = channels

        self.dense = tf.keras.layers.Dense(256 * 7 * 7, use_bias=False)
       
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))

        self.convT_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.convT_2 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', use_bias=False)
        self.convT_3 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.convT_4 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', use_bias=False)
        self.convT_5 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.convT_6 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.convT_7 = tf.keras.layers.Conv2DTranspose(self.channels, (3, 3), padding='same', use_bias=False, activation='tanh')

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.batch_norm_6 = tf.keras.layers.BatchNormalization()
        self.batch_norm_7 = tf.keras.layers.BatchNormalization()

        self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_2 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_3 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_4 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_5 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_6 = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=True):

        x = self.dense(inputs)
        x = self.batch_norm_1(x, training)

        x = self.reshape(x)

        x = self.convT_1(x)
        x = self.batch_norm_2(x, training)
        x = self.leakyrelu_1(x)

        x = self.convT_2(x)
        x = self.batch_norm_3(x, training)
        x = self.leakyrelu_2(x)

        x = self.convT_3(x)
        x = self.batch_norm_3(x, training)
        x = self.leakyrelu_3(x)

        x = self.convT_4(x)
        x = self.batch_norm_5(x, training)
        x = self.leakyrelu_4(x)

        x = self.convT_5(x)
        x = self.batch_norm_6(x, training)
        x = self.leakyrelu_5(x)

        x = self.convT_6(x)
        x = self.batch_norm_7(x, training)
        x = self.leakyrelu_6(x)

        return self.convT_7(x)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.conv_4 = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.out = tf.keras.layers.Dense(1)

        self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_2 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_3 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_4 = tf.keras.layers.LeakyReLU()

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv_1(inputs)
        x = self.leakyrelu_1(x)

        x = self.conv_2(x)
        x = self.batch_norm_1(x, training)
        x = self.leakyrelu_2(x)

        x = self.conv_3(x)
        x = self.batch_norm_2(x, training)
        x = self.leakyrelu_3(x)

        x = self.conv_4(x)
        x = self.batch_norm_3(x, training)
        x = self.leakyrelu_4(x)

        x = self.flatten(x)

        return self.out(x)


if __name__ == "__main__":
    pass

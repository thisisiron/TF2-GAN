import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, channels=1, method='transpose'):
        super(Generator, self).__init__()
        self.channels = channels
        self.method = method

        self.dense = tf.keras.layers.Dense(256 * 7 * 7, use_bias=False)
       
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))

        if self.method == 'transpose':
            self.convT_1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), padding='same', use_bias=False)
            self.convT_2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
            self.convT_3 = tf.keras.layers.Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        elif self.method == 'upsample':
            self.conv_1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False)
            self.upsample2d_1 = tf.keras.layers.UpSampling2D()
            self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False)
            self.upsample2d_2 = tf.keras.layers.UpSampling2D()
            self.conv_3 = tf.keras.layers.Conv2D(self.channels, (3, 3), padding='same', use_bias=False, activation='tanh')

        self.batch_nomal_1 = tf.keras.layers.BatchNormalization()
        self.batch_nomal_2 = tf.keras.layers.BatchNormalization()
        self.batch_nomal_3 = tf.keras.layers.BatchNormalization()

        self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_2 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_3 = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=True):

        if self.method == 'transpose':
            x = self.dense(inputs)
            x = self.batch_nomal_1(x, training)
            x = self.leakyrelu_1(x)

            x = self.reshape(x)

            x = self.convT_1(x)
            x = self.batch_nomal_2(x, training)
            x = self.leakyrelu_2(x)

            x = self.convT_2(x)
            x = self.batch_nomal_3(x, training)
            x = self.leakyrelu_3(x)

            return self.convT_3(x)

        elif self.method == 'upsample':
            # Replace Conv2DTranspose with UpSampling2D & Conv2D
            
            x = self.dense(inputs)
            x = self.batch_nomal_1(x, training)
            x = self.leakyrelu_1(x)

            x = self.reshape(x)

            x = self.conv_1(x)
            x = self.batch_nomal_2(x, training)
            x = self.leakyrelu_2(x)

            x = self.upsample2d_1(x)
            x = self.conv_2(x)
            x = self.batch_nomal_3(x, training)
            x = self.leakyrelu_3(x)

            x = self.upsample2d_2(x)
            return self.conv_3(x)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.out = tf.keras.layers.Dense(1)

        self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_2 = tf.keras.layers.LeakyReLU()

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, training=True):
        x = self.conv_1(inputs)
        x = self.leakyrelu_1(x)
        x = self.dropout_1(x, training)
 
        x = self.conv_2(x)
        x = self.leakyrelu_2(x)
        x = self.dropout_2(x, training)

        x = self.flatten(x)

        return self.out(x)


if __name__ == "__main__":
    pass

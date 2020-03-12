import tensorflow as tf  # TF 2.0
import tensorflow_addons as tfa


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

    def build_conv_block(self, dim, norm_layer, use_bias):
        conv_block = []
        p = 'same'

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias), 
                       norm_layer(), 
                       tf.keras.layers.Activation('relu')]

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias), 
                       norm_layer()]

        return tf.keras.Sequential(conv_block)

    def call(self, x, training=True):
        out = x + self.conv_block(x, training=training)
        return out


class Generator(tf.keras.layers.Layer):
    def __init__(self, 
                 channels=3,
                 ngf=64,
                 norm_layer=tfa.layers.normalizations.InstanceNormalization,
                 use_bias=False,
                 n_blocks=6):

        super(Generator, self).__init__()
        
        self.concat = tf.keras.layers.Concatenate()

        assert(n_blocks >= 0)

        model = [tf.keras.layers.Conv2D(ngf, (7, 7), padding='same', use_bias=use_bias),
                 norm_layer(),
                 tf.keras.layers.Activation('relu')]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [tf.keras.layers.Conv2D(ngf * mult * 2, (4, 4), strides=(2, 2), padding='same'),
                      norm_layer(),
                      tf.keras.layers.Activation('relu')]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [tf.keras.layers.Conv2DTranspose(int(ngf * mult / 2),
                                                      (4, 4), strides=(2, 2),
                                                      padding='same'),
                      norm_layer(),
                      tf.keras.layers.Activation('relu')]

        model += [tf.keras.layers.Conv2D(channels, (7, 7), padding='same'),
                  tf.keras.layers.Activation('tanh')]

        self.model = tf.keras.Sequential(model)

    def call(self, inputs, domains, training=True):
        domains = tf.reshape(domains, (-1, 1, 1, domains.shape[1]))
        domains = tf.tile(domains, tf.constant([1, inputs.shape[1], inputs.shape[2], 1]))
        x = self.concat([inputs, domains])
        return self.model(x, training=training)


class Discriminator(tf.keras.layers.Layer):
    """Defines a PatchGAN discriminator"""

    def __init__(self, img_shape, nd, ndf=64, n_layers=5, use_bias=False):
        super(Discriminator, self).__init__()

        h, w = img_shape[0:2]

        kw = 4
        model = [tf.keras.layers.Conv2D(ndf, (kw, kw), strides=(2, 2), padding='same'),
                 tf.keras.layers.LeakyReLU(0.01)]

        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult = 2 ** n
            model += [
                tf.keras.layers.Conv2D(ndf * nf_mult, (kw, kw), strides=(2, 2), padding='same'),
                tf.keras.layers.LeakyReLU(0.01)
            ]

        self.model = tf.keras.Sequential(model)

        self.src = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)
        
        self.cls = tf.keras.layers.Conv2D(nd, (h // 64, w // 64), strides=(1, 1), padding='valid', use_bias=use_bias)

    def call(self, inputs, training=True):
        x = self.model(inputs, training=training)
        return self.src(x), tf.squeeze(self.cls(x))

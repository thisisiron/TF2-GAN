import tensorflow as tf  # TF 2.0
import tensorflow_addons as tfa


class PixelShuffler(tf.keras.layers.Layer):
  def __init__(self, block_size):
    super(PixelShuffler, self).__init__()
    self.block_size = block_size 

  def call(self, x):
    return tf.nn.depth_to_space(x, self.block_size)


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_bias)

    def build_conv_block(self, dim, use_bias):
        conv_block = []
        p = 'same'

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias), 
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.PReLU()]

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias), 
                       tf.keras.layers.BatchNormalization()]

        return tf.keras.Sequential(conv_block)

    def call(self, x, training=True):
        out = x + self.conv_block(x, training=training)
        return out


class Generator(tf.keras.Model):
    def __init__(self, 
                 channels=3,
                 ngf=64,
                 use_bias=False,
                 n_blocks=16):

        super(Generator, self).__init__()
        n_upsampling = 2
        
        assert(n_blocks >= 0)

        init_block  = [tf.keras.layers.Conv2D(ngf, (9, 9), padding='same', use_bias=use_bias),
                       tf.keras.layers.PReLU()]
        self.init_block = tf.keras.Sequential(init_block)

        resblock = []
        for i in range(n_blocks):
            resblock += [ResnetBlock(ngf)]
        self.resblock = tf.keras.Sequential(resblock)

        model = [tf.keras.layers.Conv2D(ngf, (3, 3), padding='same', use_bias=use_bias),
                 tf.keras.layers.PReLU()]

        for i in range(n_upsampling):
            model += [tf.keras.layers.Conv2D(ngf * 4, (3, 3), padding='same'),
                      PixelShuffler(2),
                      # tf.keras.layers.UpSampling2D((2, 2)),  # Mode collapse occurred.
                      tf.keras.layers.PReLU()]

        model += [tf.keras.layers.Conv2D(channels, (9, 9), padding='same'),
                  tf.keras.layers.Activation('tanh')]

        self.model = tf.keras.Sequential(model)

    def call(self, inputs, training=True):
        x_init = self.init_block(inputs, training=training)
        x = self.resblock(x_init, training=training)
        x = x + x_init
        return self.model(x, training=training)


class Discriminator(tf.keras.Model):
    def __init__(self, ndf=64, n_layers=4, use_bias=False):
        super(Discriminator, self).__init__()

        kw = 3
        model = [tf.keras.layers.Conv2D(ndf, (kw, kw), padding='same'),
                 tf.keras.layers.LeakyReLU(0.2)]

        n = 0
        nf_mult = 2 ** n
        model += [tf.keras.layers.Conv2D(ndf * nf_mult, (kw, kw), strides=(2, 2), padding='same'),
                  tf.keras.layers.BatchNormalization(),
                  tf.keras.layers.LeakyReLU(0.2)]

        for n in range(1, n_layers + 1):
            if n % 2:  # odd
                nf_mult = 2 ** n
                s = 1
            else:
                s = 2
            model += [tf.keras.layers.Conv2D(ndf * nf_mult, (kw, kw), strides=(s, s), padding='same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.LeakyReLU(0.2)]
        
        model += [tf.keras.layers.GlobalAveragePooling2D(),
                  tf.keras.layers.Dense(1024),
                  tf.keras.layers.LeakyReLU(0.2),
                  tf.keras.layers.Dense(1),
                  tf.keras.layers.Activation('sigmoid')]

        self.model = tf.keras.Sequential(model)

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)


if __name__ == '__main__':
    g = Generator()
    g.save_weights('./model/g')
    g.load_weights('./model/g')

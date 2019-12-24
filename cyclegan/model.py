import tensorflow as tf  # TF 2.0


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, shape):
        return (shape[0], shape[1] + 2 * self.padding[0], shape[2] + 2 * self.padding[1], shape[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.norm_layer = norm_layer
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 'valid'
        if padding_type == 'reflect':
            conv_block += [ReflectionPadding2D((1, 1))]
        elif padding_type == 'zero':
            p = 'same'
        else:
            raise NotImplementedError('padding {} is not implemented'.format(padding_type))

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias), norm_layer(), tf.keras.layers.Activation('relu')]
        if use_dropout:
            conv_block += [tf.keras.layers.Dropout(0.5)]

        p = 'valid'
        if padding_type == 'reflect':
            conv_block += [ReflectionPadding2D((1, 1))]
        elif padding_type == 'zero':
            p = 'same'
        else:
            raise NotImplementedError('padding {} is not implemented'.format(padding_type))

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias), norm_layer()]

        return tf.keras.Sequential(conv_block)

    def call(self, x, training=True):
        out = x + self.conv_block(x, training=training)
        return out


class Generator(tf.keras.Model):
    def __init__(self, 
                 output_dim=3,
                 ngf=64,
                 norm_layer=tf.keras.layers.BatchNormalization,
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):

        super(Generator, self).__init__()

        assert(n_blocks >= 0)

        use_bias = norm_layer == tf.keras.layers.BatchNormalization

        model = [ReflectionPadding2D((3, 3)),
                 tf.keras.layers.Conv2D(ngf, (7, 7), padding='valid', use_bias=use_bias),
                 norm_layer(),
                 tf.keras.layers.Activation('relu')]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [tf.keras.layers.Conv2D(ngf * mult * 2, (3, 3), strides=(2, 2), padding='same', use_bias=use_bias),
                      norm_layer(),
                      tf.keras.layers.Activation('relu')]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [tf.keras.layers.Conv2DTranspose(int(ngf * mult / 2),
                                                      (3, 3), strides=(2, 2),
                                                      padding='same', output_padding=1,
                                                      use_bias=use_bias),
                      norm_layer(),
                      tf.keras.layers.Activation('relu')]
        model += [ReflectionPadding2D((3, 3)),
                  tf.keras.layers.Conv2D(output_dim, kernel_size=7, padding='valid'),
                  tf.keras.layers.Activation('tanh')]

        self.model = tf.keras.Sequential(model)

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)


class Discriminator(tf.keras.layers.Layer):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ndf=64, n_layers=3, norm_layer=tf.keras.layers.BatchNormalization):
        super(Discriminator, self).__init__()

        use_bias = norm_layer == tf.keras.layers.BatchNormalization

        kw = 4
        sequence = [tf.keras.layers.Conv2D(ndf, (kw, kw), strides=(2, 2), padding='same'),
                    tf.keras.layers.LeakyReLU(0.2)]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            sequence += [
                tf.keras.layers.Conv2D(ndf * nf_mult, (kw, kw), strides=(2, 2), padding='same', use_bias=use_bias),
                norm_layer(),
                tf.keras.layers.LeakyReLU(0.2)
            ]

        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            tf.keras.layers.Conv2D(ndf * nf_mult, (kw, kw), strides=(1, 1), padding='same', use_bias=use_bias),
            norm_layer(),
            tf.keras.layers.LeakyReLU(0.2)
        ]

        sequence += [tf.keras.layers.Conv2D(1, (kw, kw), strides=(1, 1), padding='same')]
        self.model = tf.keras.Sequential(sequence)

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)


if __name__ == "__main__":
    generator = Generator(3)
    discriminator = Discriminator()
    pass

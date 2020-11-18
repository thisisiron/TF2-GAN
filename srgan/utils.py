import cv2
import tensorflow as tf


IMG_H_SIZE = 256
IMG_W_SIZE = 256


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def total_variation_loss(x):    
    h, w = x.shape[1], x.shape[2]
    a = tf.math.square(x[:, :h- 1, :w - 1, :] - x[:, 1:, :w - 1, :])    
    b = tf.math.square(x[:, :h - 1, :w - 1, :] - x[:, :w - 1, 1:, :])    
    return tf.math.reduce_sum(tf.math.pow(a + b, 1.25))


def content_loss(loss_object, hr_feat, sr_feat):
    total_loss = loss_object(hr_feat, sr_feat)
    return total_loss# * 0.006


def vgg_layers(layer_name):
    """ Creates a vgg model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    output = vgg.get_layer(layer_name).output
    model = tf.keras.Model(vgg.input, output)
    return model


class ContentModel(tf.keras.models.Model):
    def __init__(self, content_layer):
        super(ContentModel, self).__init__()
        self.vgg = vgg_layers(content_layer)
        self.content_layer = content_layer
        self.vgg.trainable = False

    @tf.function
    def call(self, inputs):
        "Expects float input in [-1, 1]"
        inputs = (inputs + 1) * 127.5
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)  # Must use "tf.function"
        content_output = self.vgg(preprocessed_input)
        return content_output


def save_imgs(epoch, generator, lr_images, hr_images):
    gen_imgs = generator(lr_images, training=False)

    for i in range(gen_imgs.shape[0]):
        cv2.imwrite('./images/sr_{}_{}.png'.format(epoch, i), (gen_imgs[i].numpy()[..., ::-1] + 1) * 127.5)
        cv2.imwrite('./images/hr_{}_{}.png'.format(epoch, i), (hr_images[i].numpy()[..., ::-1] + 1) * 127.5)
        cv2.imwrite('./images/lr_{}_{}.png'.format(epoch, i), (lr_images[i].numpy()[..., ::-1] + 1) * 127.5)

        resized = cv2.resize((lr_images[i].numpy()[..., ::-1] + 1) * 127.5, (IMG_W_SIZE, IMG_H_SIZE))
        cv2.imwrite('./images/re_{}_{}.png'.format(epoch, i), resized)


def preprocess_data(file_path, ratio=4):
    image = process_path(file_path)
    resized_image = resize(image, (IMG_H_SIZE//ratio, IMG_W_SIZE//ratio))
    image = resize(image, (IMG_H_SIZE, IMG_W_SIZE))
    image = normalize(image)
    resized_image = normalize(resized_image)

    return resized_image, image


def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, size):
    h, w = size
    image = tf.image.resize(image, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    return img

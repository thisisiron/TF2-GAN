import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def cycle_loss(loss_object, real_image, cycled_image, _lambda=10):
    return loss_object(real_image, cycled_image) * _lambda


def identity_loss(loss_object, real_image, same_image, _lambda=10):
    return loss_object(real_image, same_image) * 0.5 * _lambda


def save_imgs(epoch, generator, real_x):
    gene_imgs = generator(real_x, training=False)

    gene_imgs = ((gene_imgs.numpy() + 1) * 127.5).astype(np.uint8)
    real_x = ((real_x.numpy() + 1) * 127.5).astype(np.uint8)

    fig = plt.figure(figsize=(8, 16))

    tmp = 0
    for i in range(0, real_x.shape[0]):
        plt.subplot(4, 2, i + 1 + tmp)
        plt.imshow(real_x[i])
        plt.axis('off')
        plt.subplot(4, 2, i + 2 + tmp)
        plt.imshow(gene_imgs[i])
        plt.axis('off')
        tmp += 1

    fig.savefig("images/result_{}.png".format(str(epoch).zfill(5)))
    print('Success saving images')


def normalize_img(x, dtype):
    x = tf.cast(x, dtype=dtype)
    return x / 127.5 - 1  # range: -1 ~ 1


def decode_img(img):
    img = tf.image.decode_jpeg(img)
    img = normalize_img(img, tf.float32)  # range: -1 ~ 1
    return img


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img


def augment_image(img, mask):
    rand = tf.random.uniform(())
    if rand <= 0.25:
        img = flip(img)
        mask = flip(mask)
    elif rand > 0.25 and rand <= 0.5:
        d = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = rotate(img, d)
        mask = rotate(mask, d)
    elif rand > 0.5 and rand <= 0.75:
        img = color(img)

    return img, mask


def flip(img):
    img = tf.image.flip_left_right(img)
    img = tf.image.flip_up_down(img)
    return img


def rotate(img, degree):
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(img, degree)


def color(img):
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img

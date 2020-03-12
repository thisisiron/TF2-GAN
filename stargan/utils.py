import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np


def gradient_penalty_loss(averaged_output, x_hat):
    gradients = tf.gradients(averaged_output, x_hat)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)

    gradient_penalty = tf.square(gradients_l2_norm - 1)

    return tf.reduce_mean(gradient_penalty)


def discriminator_loss(real_output, fake_output, averaged_output, interpolated_img, lamb_gp=10):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    gp_loss = gradient_penalty_loss(averaged_output, interpolated_img)
    total_loss = real_loss + fake_loss + gp_loss* lamb_gp
    return total_loss


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def reconstrution_loss(loss_object, real_image, recon_image, lamb_rec=10):
    return loss_object(real_image, recon_image) * lamb_rec 


def domain_classification_loss(loss_object, category, output, lamb_cls=1):
    return loss_object(category, output) * lamb_cls


def random_weighted_average(inputs):
    alpha = tf.random.uniform((inputs[0].shape[0], 1, 1, 1))
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def save_imgs(epoch, generator, real_x):
    gene_imgs = generator(real_x, [0, 1, 0, 1, 0], training=False)

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


def preprocess_data(file_path, image_label, target_label):
    # image = tf.io.read_file(file_path)
    # image = tf.image.decode_jpeg(image)
    image = process_path(file_path)
    image = resize(image, (128, 128))
    image = normalize(image)

    return image, image_label, target_label 


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

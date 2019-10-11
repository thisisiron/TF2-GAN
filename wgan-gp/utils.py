import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def gradient_penalty_loss(y_pred, y_average):
    gradients = tf.gradients(y_pred, y_average)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)

    gradient_penalty = tf.square(gradients_l2_norm)

    return tf.reduce_mean(gradient_penalty)


def discriminator_loss(real_output, generated_output, validity_interpolated, interpolated_img, lambda_=10):

    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(generated_output)
    gp_loss = gradient_penalty_loss(validity_interpolated, interpolated_img)

    return real_loss + fake_loss + gp_loss * lambda_ 


def generator_loss(generated_output):
    return -tf.reduce_mean(generated_output)


def random_weighted_average(inputs):
    alpha = tf.random.uniform((inputs[0].shape[0], 1, 1, 1))
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image


def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise, False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

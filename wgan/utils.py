import tensorflow as tf
import matplotlib.pyplot as plt


def discriminator_loss(real_output, generated_output):

    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(generated_output)

    return real_loss + fake_loss


def generator_loss(generated_output):
    return -tf.reduce_mean(generated_output)


def save_imgs(epoch, generator, noise):

    gen_imgs = generator(noise, False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

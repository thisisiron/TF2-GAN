import tensorflow as tf
import matplotlib.pyplot as plt


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, height, width):
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def convert_onehot(label, num_classes):
    return tf.one_hot(label, num_classes)


def preprocess_image(data, img_shape, num_classes):
    image = data['image']
    image = resize(image, img_shape[0], img_shape[1])
    image = normalize(image)

    label = convert_onehot(data['label'], num_classes)

    return image, label


def save_imgs(epoch, generator, noise):
    # You can set it you want.
    labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 12
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 13
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 14
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 15
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 16
              ]
    labels = tf.convert_to_tensor(labels)
    gen_imgs = generator(noise, labels, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

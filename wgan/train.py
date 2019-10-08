import os
import time

import numpy as np
import tensorflow as tf  # TF 2.0

from model import Generator, Critic
from utils import discriminator_loss, generator_loss, save_imgs


def train():

    (train_data, _), (_, _) = tf.keras.datasets.mnist.load_data()

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100
    epochs = 800
    batch_size = 32
    buffer_size = 6000
    save_interval = 50
    n_critic = 5

    generator = Generator()
    discriminator = Critic()

    gen_optimizer = tf.keras.optimizers.RMSprop(0.00005)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.00005)

    # Rescale -1 to 1
    train_data = train_data / 127.5 - 1.
    train_data = np.expand_dims(train_data, axis=3).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size).batch(batch_size)

    @tf.function
    def train_discriminator(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as disc_tape:
            generated_imgs = generator(noise, training=True)

            generated_output = discriminator(generated_imgs, training=True)
            real_output = discriminator(images, training=True)

            disc_loss = discriminator_loss(real_output, generated_output)

        grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        for param in discriminator.trainable_variables:
            # Except gamma and beta in Batch Normalization
            if param.name.split('/')[-1].find('gamma') == -1 and param.name.split('/')[-1].find('beta') == -1:
                param.assign(tf.clip_by_value(param, -0.01, 0.01))

        return disc_loss

    @tf.function
    def train_generator():
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as gen_tape: 
            generated_imgs = generator(noise, training=True)
            generated_output = discriminator(generated_imgs, training=True)

            gen_loss = generator_loss(generated_output)

        grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

        return gen_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        disc_loss = 0
        gen_loss = 0

        for images in train_dataset:
            disc_loss += train_discriminator(images)

            if disc_optimizer.iterations.numpy() % n_critic == 0:
                gen_loss += train_generator()

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, gen_loss / batch_size, disc_loss / (batch_size*n_critic)))

        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()

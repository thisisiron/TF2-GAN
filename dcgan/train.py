import os
import time

import tensorflow as tf  # TF 2.0
import tensorflow_datasets as tfds

from model import Generator, Discriminator
from utils import generator_loss, discriminator_loss, save_imgs, normalize


def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='/data/tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100
    epochs = 800
    batch_size = 200
    buffer_size = 6000
    save_interval = 50

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise)

            real_output = discriminator(images)
            generated_output = discriminator(generated_images)

            gen_loss = generator_loss(cross_entropy, generated_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()

import time
import tensorflow as tf

from model import Generator, Discriminator
from utils import generator_loss, discriminator_loss 

import numpy as np

import os


def train():
    (train_data, _), (_, _) = tf.keras.datasets.mnist.load_data()

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

    # Rescale -1 to 1
    train_data = train_data / 127.5 - 1.
    train_data = np.expand_dims(train_data, axis=3).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise)
            real_output = discriminator(images)
            generated_output = discriminator(generated_images)

            gen_loss = generator_loss(cross_entropy, generated_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss


    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

        total_gen_loss += gen_loss
        total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss, total_disc_loss))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, latent_dim)


if __name__ == "__main__":
    train()

import os
import time
import cv2

from glob import glob

import tensorflow as tf  # TF 2.0
import tensorflow_addons as tfa

from model import Generator, Discriminator
from utils import generator_loss, discriminator_loss
from utils import cycle_loss, identity_loss
from utils import process_path, augment_image, save_imgs


AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 2000


def train():
    # settting hyperparameter
    epochs = 800
    batch_size = 4
    save_interval = 20

    train_A = tf.data.Dataset.list_files('./datasets/monet2photo/train/A/*')
    train_B = tf.data.Dataset.list_files('./datasets/monet2photo/train/B/*')

    train_A = train_A.map(lambda x: process_path(x), num_parallel_calls=AUTOTUNE).repeat(1)
    train_B = train_B.map(lambda x: process_path(x), num_parallel_calls=AUTOTUNE).repeat(1)

    train_dataset = tf.data.Dataset.zip((train_A, train_B)).map(lambda x, y: augment_image(x, y), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.cache().batch(batch_size).shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    if not os.path.exists('./images'):
        os.makedirs('./images')

    gene_G = Generator(norm_layer=tfa.layers.InstanceNormalization)
    gene_F = Generator(norm_layer=tfa.layers.InstanceNormalization)
    disc_X = Discriminator(norm_layer=tfa.layers.InstanceNormalization)
    disc_Y = Discriminator(norm_layer=tfa.layers.InstanceNormalization)

    gen_g_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    gen_f_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_x_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_y_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mae = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_step(real_x, real_y):

        with tf.GradientTape(persistent=True) as tape:
            fake_y = gene_G(real_x)
            rec_x = gene_F(fake_y)

            fake_x = gene_F(real_y)
            rec_y = gene_G(fake_x)

            same_x = gene_G(real_x)
            same_y = gene_F(real_y)

            disc_real_x = disc_X(real_x)
            disc_real_y = disc_Y(real_y)

            disc_fake_x = disc_X(fake_x)
            disc_fake_y = disc_Y(fake_y)

            # Loss Func.
            disc_x_loss = discriminator_loss(cross_entropy, disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(cross_entropy, disc_real_y, disc_fake_y)

            gene_g_loss = generator_loss(cross_entropy, disc_fake_y)
            gene_f_loss = generator_loss(cross_entropy, disc_fake_x)

            cycle_x_loss = cycle_loss(mae, real_x, rec_x)
            cycle_y_loss = cycle_loss(mae, real_y, rec_y,)

            total_cycle_loss = cycle_x_loss + cycle_y_loss
            total_gen_g_loss = gene_g_loss + total_cycle_loss + identity_loss(mae, real_y, same_y)
            total_gen_f_loss = gene_f_loss + total_cycle_loss + identity_loss(mae, real_x, same_x)
            
        grad_gen_G = tape.gradient(total_gen_g_loss, gene_G.trainable_variables)
        grad_gen_F = tape.gradient(total_gen_f_loss, gene_F.trainable_variables)

        grad_disc_X = tape.gradient(disc_x_loss, disc_X.trainable_variables)
        grad_disc_Y = tape.gradient(disc_y_loss, disc_Y.trainable_variables)

        gen_g_optimizer.apply_gradients(zip(grad_gen_G, gene_G.trainable_variables))
        gen_f_optimizer.apply_gradients(zip(grad_gen_F, gene_F.trainable_variables))

        disc_x_optimizer.apply_gradients(zip(grad_disc_X, disc_X.trainable_variables))
        disc_y_optimizer.apply_gradients(zip(grad_disc_Y, disc_Y.trainable_variables))

        return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss

    print('Training...')

    for epoch in range(epochs):
        start = time.time()
        total_gen_g_loss = 0
        total_gen_f_loss = 0
        total_disc_x_loss = 0
        total_disc_y_loss = 0

        for images_x, images_y in train_dataset:
            gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = train_step(images_x, images_y)

            total_gen_g_loss += gen_g_loss
            total_gen_f_loss += gen_f_loss
            total_disc_x_loss += disc_x_loss
            total_disc_y_loss += disc_y_loss

        print("""Time for epoch {} is {} sec - 
        G gen_loss = {}, F gen_loss = {} D x_loss = {}, D y_loss = {}""".format(epoch + 1,
                                                                                time.time() - start,
                                                                                total_gen_g_loss / batch_size,
                                                                                total_gen_f_loss / batch_size,
                                                                                total_disc_x_loss / batch_size,
                                                                                total_disc_y_loss / batch_size,))

        if epoch % save_interval == 0:
            save_imgs(epoch, gene_F, images_y)


if __name__ == "__main__":
    train()

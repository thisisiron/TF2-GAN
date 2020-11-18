import os
import time

import tensorflow as tf  # TF 2.0
import tensorflow_datasets as tfds

from model import Generator, Discriminator
from utils import save_imgs
from utils import preprocess_data 
from utils import ContentModel 
from utils import generator_loss
from utils import discriminator_loss 
from utils import content_loss
from utils import total_variation_loss


AUTOTUNE = tf.data.experimental.AUTOTUNE


def train():
    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    epochs = 800
    batch_size = 4 
    save_interval = 2 

    # Model setting
    generator = Generator(n_blocks=5)
    discriminator = Discriminator()

    # Optimizer setting
    gen_optimizer = tf.keras.optimizers.Adam(0.0005, 0.9)
    disc_optimizer = tf.keras.optimizers.Adam(0.0005, 0.9)

    image_ds = tf.data.Dataset.list_files('./data/data512x512/*', shuffle=True)
    train_dataset = image_ds.map(lambda x: preprocess_data(x), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE).cache()

    # Loss setting
    content_layer = 'block5_conv4'  # SRGAN-VGG54
    extractor = ContentModel(content_layer)
    extractor.trainable = False

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(lr_images, hr_images):
        with tf.GradientTape(persistent=True) as tape:
            sr_images = generator(lr_images)  # sr -> super resolution

            real_output = discriminator(hr_images)
            fake_output = discriminator(sr_images)

            # adversarial loss
            gen_loss = generator_loss(cross_entropy, fake_output) * 1e-3
            disc_loss = discriminator_loss(cross_entropy, real_output, fake_output) * 1e-3

            # content loss
            hr_feat = extractor(hr_images)
            sr_feat = extractor(sr_images)
            cont_loss = content_loss(mse, hr_feat, sr_feat) * 0.006

            perc_loss = cont_loss + gen_loss

        grad_gen = tape.gradient(perc_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss, cont_loss

    total_iter = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0
        total_cont_loss = 0

        for i, (lr_images, hr_images) in enumerate(train_dataset, 1):
            total_iter += 1
            gen_loss, disc_loss, cont_loss = train_step(lr_images, hr_images)

            if i % 100 == 0:
                print(f'Batch:{i}({total_iter}) -> gen_loss: {gen_loss}, disc_loss: {disc_loss}, cont_loss: {cont_loss}')
                save_imgs(epoch, generator, lr_images, hr_images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            total_cont_loss += cont_loss

        print('Time for epoch {} is {} sec -> gen_loss: {}, disc_loss: {}, cont_loss: {}'.format(epoch, 
                                                                                                 time.time() - start, 
                                                                                                 total_gen_loss / i, 
                                                                                                 total_disc_loss / i,
                                                                                                 total_cont_loss / i))


if __name__ == "__main__":
    train()

import time
import tensorflow as tf
from model import Generator, Discriminator
from utils import generator_loss, discriminator_loss, create_optimizer, generate_and_save_images
from IPython import display


def train():
    (train_data, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # settting hyperparameter
    latent_dim = 100
    epochs = 50
    batch_size = 150
    buffer_size = 6000

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = create_optimizer()
    disc_optimizer = create_optimizer()

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
    train_data = (train_data - 127.5)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size).batch(batch_size)

    seed = tf.random.normal([16, latent_dim])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise)
            real_output = discriminator(images)
            generated_output = discriminator(generated_images)

            gen_loss = generator_loss(generated_output)
            disc_loss = discriminator_loss(real_output, generated_output)

        grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(epochs):
        start = time.time()
        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

        print('Time for epoch {} is {} sec: gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, gen_loss, disc_loss))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


if __name__ == "__main__":
    train()

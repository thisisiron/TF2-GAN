import os
import time

import tensorflow as tf  # TF 2.0
from glob import glob
import numpy as np
import cv2

from model import Generator, Discriminator
from utils import generator_loss, discriminator_loss, save_imgs, preprocess_data 
from utils import domain_classification_loss, reconstrution_loss
from utils import random_weighted_average


AUTOTUNE = tf.data.experimental.AUTOTUNE


def train():
    ATTRIBUTES = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    with open('./data/celeba/list_attr_celeba.txt', 'r') as f:
        attr_info = f.readlines()

    attr2idx = {}
    idx2attr = {}
    for i, attr in enumerate(attr_info[1].split(' ')):  # attr_info[1] -> Header
        if attr in ATTRIBUTES:
            attr2idx[attr] = i
            idx2attr[i] = attr

    # Setting Label Inofor.
    image_label = [[0. for _ in range(len(ATTRIBUTES))] for _ in range(len(attr_info[2:]))]
    for i, attr in enumerate(attr_info[2:]):
        cols = attr.split()
        attr_value = cols[1:]
        for j, attr in enumerate(ATTRIBUTES):
            if attr_value[attr2idx[attr]] == '1':
                image_label[i][j] = 1.

    # Settting hyperparameter
    epochs = 800
    save_interval = 2000
    n_critic = 5
    batch_size = 64 
    img_shape = (128, 128, 3)
    num_class = 5

    image_ds = tf.data.Dataset.list_files('./data/celeba/images/*', shuffle=False)
    ori_label_ds = tf.data.Dataset.from_tensor_slices(image_label)
    tar_label_ds = tf.data.Dataset.from_tensor_slices(image_label).shuffle(30000)

    train_dataset = tf.data.Dataset.zip((image_ds, ori_label_ds, tar_label_ds)).map(lambda x, y, z: preprocess_data(x, y, z), num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    # train_dataset = tf.data.Dataset.zip((train_dataset, tar_label_ds)).batch(batch_size).prefetch(AUTOTUNE)
    print('loaded dataset')

    if not os.path.exists('./images'):
        os.makedirs('./images')

    generator = Generator()
    discriminator = Discriminator(img_shape, num_class)

    gen_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)

    # loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    l1 = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_generator(images, ori_labels, tar_labels):
        # gen_ori_labels = tf.reshape(ori_labels, (-1, 1, 1, ori_labels.shape[1]))
        # gen_ori_labels = tf.tile(gen_ori_labels, tf.constant([1, images.shape[1], images.shape[2], 1]))

        # gen_tar_labels = tf.reshape(tar_labels, (-1, 1, 1, tar_labels.shape[1]))
        # gen_tar_labels = tf.tile(gen_tar_labels, tf.constant([1, images.shape[1], images.shape[2], 1]))

        with tf.GradientTape(persistent=True) as tape:
            # fake_images = generator(images, gen_tar_labels)
            # recon_images = generator(fake_images, gen_ori_labels)
            fake_images = generator(images, tar_labels)
            recon_images = generator(fake_images, ori_labels)

            fake_output, fake_class = discriminator(fake_images)

            gen_loss = generator_loss(fake_output)
            fake_class_loss = domain_classification_loss(bce, tar_labels, fake_class)
            recon_loss = reconstrution_loss(l1, images, recon_images) 

            total_gen_loss = gen_loss + fake_class_loss + recon_loss 

        grad_gen = tape.gradient(total_gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

        return fake_images, gen_loss, fake_class_loss, recon_loss 

    @tf.function
    def train_discriminator(images, ori_labels, tar_labels):
        # gen_tar_labels = tf.reshape(tar_labels, (-1, 1, 1, tar_labels.shape[1]))
        # gen_tar_labels = tf.tile(gen_tar_labels, tf.constant([1, images.shape[1], images.shape[2], 1]))

        with tf.GradientTape(persistent=True) as tape:
            # real
            real_output, real_class = discriminator(images)

            # fake
            fake_images = generator(images, tar_labels)
            fake_output, fake_class = discriminator(fake_images)

            # x_hat
            interpolated_img = random_weighted_average([images, fake_images])
            averaged_output, _ = discriminator(interpolated_img)

            disc_loss = discriminator_loss(real_output, fake_output, averaged_output, interpolated_img)

            real_class_loss = domain_classification_loss(bce, ori_labels, real_class)

            total_disc_loss = disc_loss + real_class_loss

        grad_disc = tape.gradient(total_disc_loss, discriminator.trainable_variables)

        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return real_class_loss, disc_loss

    print('Training...')
    for epoch in range(epochs):
        start = time.time()

        for images, ori_labels, tar_labels in train_dataset:
            real_cls_loss, disc_loss = train_discriminator(images, ori_labels, tar_labels)
            log = 'Time for epoch {}/{} is {} sec : iter {} - disc_loss = {}, real_cls_loss = {}'.format(epoch + 1, epochs, time.time() - start, disc_optimizer.iterations.numpy(), disc_loss/batch_size, real_cls_loss/batch_size)
            
            if disc_optimizer.iterations.numpy() % n_critic == 0:
                fake_images, gen_loss, fake_cls_loss, recon_loss = train_generator(images, ori_labels, tar_labels)
                log += 'gen_loss = {}, fake_cls_loss = {}, recon_loss = {}'.format(gen_loss/batch_size, fake_cls_loss/batch_size, recon_loss/batch_size)

            if disc_optimizer.iterations.numpy() % save_interval == 0:
                for idx, (orig_img, fake_img) in enumerate(zip(images, fake_images)):
                    tmp = np.asarray((orig_img.numpy() + 1) * 127.5, dtype=np.uint8)
                    cv2.imwrite('images/Step{}_Batch{}_Ori{}.png'.format(str(disc_optimizer.iterations.numpy()).zfill(6), str(idx).zfill(3), str(ori_labels[idx].numpy()).replace(' ', '')), tmp[..., ::-1])
                    tmp = np.asarray((fake_img.numpy() + 1) * 127.5, dtype=np.uint8)
                    cv2.imwrite('images/Step{}_Batch{}_Tar{}.png'.format(str(disc_optimizer.iterations.numpy()).zfill(6), str(idx).zfill(3), str(tar_labels[idx].numpy()).replace(' ', '')), tmp[..., ::-1])
                print('image saved!')

            print(log)


if __name__ == "__main__":
    train()

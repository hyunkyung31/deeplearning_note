"""
MNIST DCGAN — 클래스 베이스 버전
함수 베이스 노트북(20260707_AE_VAE_GAN.ipynb)과 동일 알고리즘, 구조만 GAN(keras.Model)로 캡슐화.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim: int) -> keras.Sequential:
    model = keras.Sequential(name="generator")
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(1, 7, padding="same", activation="tanh"))
    return model


def build_discriminator(img_shape: tuple[int, int, int]) -> keras.Sequential:
    model = keras.Sequential(name="discriminator")
    model.add(layers.Input(shape=img_shape))
    model.add(layers.Conv2D(64, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


def load_data(batch_size: int = 128) -> tf.data.Dataset:
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")
    x_train = (x_train - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)
    return (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(60000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


class GAN(keras.Model):
    def __init__(self, generator: keras.Sequential, discriminator: keras.Sequential, latent_dim: int):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, **kwargs):
        super().compile(**kwargs)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.bce(tf.ones_like(real_output) * 0.9, real_output)
        fake_loss = self.bce(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def _generator_loss(self, fake_output):
        return self.bce(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            d_loss = self._discriminator_loss(real_output, fake_output)
            g_loss = self._generator_loss(fake_output)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


def save_models(gan: GAN, prefix: str = "gan_mnist") -> None:
    gan.generator.save(f"{prefix}_generator.keras")
    gan.discriminator.save(f"{prefix}_discriminator.keras")


def load_gan(prefix: str = "gan_mnist", latent_dim: int = 100, img_shape=(28, 28, 1)) -> GAN:
    generator = keras.models.load_model(f"{prefix}_generator.keras")
    discriminator = keras.models.load_model(f"{prefix}_discriminator.keras")
    gan = GAN(generator, discriminator, latent_dim)
    gan.compile()
    return gan


def generate_from_noise(gan: GAN, noise: np.ndarray) -> np.ndarray:
    imgs = gan.generator(noise, training=False)
    return ((imgs + 1) / 2.0).numpy()


def main():
    latent_dim = 100
    img_shape = (28, 28, 1)
    epochs = 50
    save_dir = "gan_images"

    generator = build_generator(latent_dim)
    discriminator = build_discriminator(img_shape)
    gan = GAN(generator, discriminator, latent_dim)
    gan.compile()

    dataset = load_data()
    gan.fit(dataset, epochs=epochs, verbose=1)

    os.makedirs(save_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (25, latent_dim))
    images = generate_from_noise(gan, noise)
    print(f"Generated batch shape: {images.shape}")

    save_models(gan)
    print("Saved generator/discriminator .keras files")


if __name__ == "__main__":
    main()

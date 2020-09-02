import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Generator(tf.keras.models.Sequential):
    def __init__(self):
        super().__init__()
        self.create_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def create_model(self):
        self.add(tf.keras.layers.Dense(7*7*256, input_shape=(100,)))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Reshape((7, 7, 256)))

        self.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=5, strides=(1, 1), padding='same', use_bias=False))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=(2, 2), padding='same', use_bias=False))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=(2, 2), padding='same', activation='tanh'))
        self.add(tf.keras.layers.BatchNormalization())

    def loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


class Discriminator(tf.keras.models.Sequential):
    def __init__(self):
        super().__init__()
        self.create_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def create_model(self):
        self.add(tf.keras.layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding='same',
                                        input_shape=(28, 28, 1)))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding='same'))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1))

    def loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


@tf.function
def train_step(images):

    random = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(random, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated, training=True)

        gen_loss = generator.loss(fake_output)
        disc_loss = discriminator.loss(real_output, fake_output)

        gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator.optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        discriminator.optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(epochs):

    for epoch in range(epochs):

        print("epochs: ", epoch+1)
        for index, batch in enumerate(x_train):
            print("{}/{}".format(index + 1, len(x_train)))
            (gloss, dloss) = train_step(batch)
            print("generator loss: {}\ndiscriminator loss: {}".format(np.mean(gloss), np.mean(dloss)))
            manager.save()
            g = generator(random_seed).numpy().reshape((28, 28, 1))
            g = map(cv2.resize(g, (224, 224)), -1, 1, 0, 255)
            cv2.imwrite("gan_output/mnist_gan/img-epoch{}-step{}.png".format(epoch+1, index+1), g)


@tf.function
def get_sample():
    return tf.random.normal([1, 100])


def map(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
BATCH_SIZE = 256

x_train = x_train.reshape((60000, 28, 28, 1))
x_train = (x_train - 127.5) / 127.5
x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

generator = Generator()
discriminator = Discriminator()

checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, "models/mnist_gan", max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)

tf.random.set_seed(0)
random_seed = get_sample()

# train(5)

x = generator(random_seed)

pred = discriminator(x)

print(pred)

g = generator(random_seed).numpy()
g = g.reshape((28, 28, 1))
g = cv2.resize(g, (224, 224))
# g = map(g, -1, 1, 0, 255)

cv2.imshow("img", g)
cv2.waitKey(0)
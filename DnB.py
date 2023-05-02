from tensorflow import keras
import tensorflow as tf

# Needed for exporting to audio
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

#https://www.tensorflow.org/tutorials/audio/simple_audio
dataset = keras.utils.audio_dataset_from_directory(
    directory="audio",
    batch_size=6,
)

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels
  
#dataset = dataset.map(squeeze, tf.data.AUTOTUNE)

# The following gives us (32, 661500), meaning we have 32 audio files
#   per batch, and 661,500 samples per file.

sample_rate = 44100

#for example_audio, example_labels in dataset.take(1):  
#  print(example_audio)
#  print(example_audio.shape)
#  print(example_audio.dtype)
#  print(example_audio[0])
  
#for example_audio in dataset.take(1):
  
#    data = tf.audio.encode_wav(example_audio[0][0].reshape((sample_rate*15, 1)), sample_rate, name=None)
#    tf.io.write_file(f"Generated/example_audio.wav", data, name=None)
  
#exit()

from tensorflow.keras import layers

discriminator = keras.Sequential(
[
        keras.Input(shape=(661500, 1)),
 
        layers.Conv1D(128, kernel_size=5, strides=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
 
        layers.Conv1D(128, kernel_size=7, strides=5, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
 
        layers.Conv1D(128, kernel_size=5, strides=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
 
        layers.Conv1D(128, kernel_size=7, strides=5, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv1D(128, kernel_size=5, strides=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
 
        layers.Conv1D(128, kernel_size=7, strides=5, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
 
        layers.Conv1D(128, kernel_size=9, strides=7, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
 
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

print(discriminator.summary())

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(28 * 128),
        layers.Reshape((28, 128)),
     
        layers.Conv1DTranspose(128, kernel_size=9, strides=7, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1DTranspose(128, kernel_size=7, strides=5, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1DTranspose(128, kernel_size=5, strides=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1DTranspose(128, kernel_size=7, strides=5, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1DTranspose(128, kernel_size=5, strides=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1DTranspose(128, kernel_size=7, strides=5, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1DTranspose(128, kernel_size=5, strides=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
     
        layers.Conv1D(1, kernel_size=5, padding="same", activation="tanh"),
        
        # Rescale from [0, 1] (sigmoid) to [-1, 1]
        #layers.Rescaling(2, -1),
    ],
    name="generator",
)

print(generator.summary())

import tensorflow as tf

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metric(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        real_images = real_images[0]
        #print(real_images)
        #print(real_images[0])
        #print(real_images[0][0])
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        #print("Generated: ", generated_images[0])
        #print("Real: ", real_images[0])
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
        axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
            
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(),
        "g_loss": self.g_loss_metric.result()}
        
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
        shape=(self.num_img, self.latent_dim))
        generated_audio = self.model.generator(random_latent_vectors)
        #generated_images *= 255
        generated_audio.numpy()

        if epoch % 10 == 0:
            for i in range(self.num_img):
                #img = keras.utils.array_to_img(generated_images[i])
                #img.save(f"generated_img_{epoch:03d}_{i}.png")
                
                # We need np_config.enable_numpy_behavior() for the following line
                data = tf.audio.encode_wav(generated_audio[0].reshape((sample_rate*15, 1)), sample_rate, name=None)
                tf.io.write_file(f"Generated/generated_music_{epoch:03d}_{i}.wav", data, name=None)

                #with open(f"generated_music_{epoch:03d}_{i}.wav", "w") as file:
                #    file.write(data)
        
epochs = 5000

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
  #d_optimizer=keras.optimizers.Adam(learning_rate=0.00001),
  #g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
  d_optimizer=keras.optimizers.Adam(learning_rate=0.00005),
  g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
  loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
  dataset, epochs=epochs,
  callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
)
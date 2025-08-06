import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
import time

# Check if TensorFlow is using a GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs found. Running on CPU.")

# Define the path to the dataset
dataset_path = "D:/Dataset/HAM10000_images_part_2"

# Function to load and preprocess images
def load_images(image_dir, image_size=(64, 64)):
    images = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        img = img.astype('float32') / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)

# Load and preprocess the dataset
images = load_images(dataset_path)

# Define input shape
input_shape = (64, 64, 3)

# Generator Model with DCGAN architecture
def build_generator():
    model = Sequential()
    model.add(Input(shape=(100,)))
    model.add(Dense(8 * 8 * 256))  # Increase size for intermediate layer
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))  # Adjust reshape to (8, 8, 256)
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))  # Output layer
    return model

# Discriminator Model with DCGAN architecture
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=(64, 64, 3)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile GAN model with DCGAN architecture
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    
    gan_input = Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    return gan

generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Function to generate random noise
def generate_noise(batch_size, noise_dim=100):
    return np.random.normal(0, 1, (batch_size, noise_dim))

# Training the GAN with cooldown periods
def train_gan(generator, discriminator, gan, images, epochs=20000, batch_size=32):
    cooldown_time = 300  # 7 minutes = 420 seconds

    for epoch in range(1, epochs + 1):
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]

        noise = generate_noise(batch_size)
        generated_images = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = generate_noise(batch_size)
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Discriminator Loss: {d_loss[0]} | Generator Loss: {g_loss}")

            print(f"Cooling down for {cooldown_time // 60} minutes...")
            time.sleep(cooldown_time)

    print("Training complete!")

# Start training
train_gan(generator, discriminator, gan, images)

# Save the final models after training
final_generator_file = "final_generator.h5"
final_discriminator_file = "final_discriminator.h5"

generator.save(final_generator_file)
discriminator.save(final_discriminator_file)

print(f"Final generator saved as {final_generator_file}")
print(f"Final discriminator saved as {final_discriminator_file}")

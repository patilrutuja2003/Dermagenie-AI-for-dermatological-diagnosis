import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Path to the dataset folder
image_folder_path = "D:/Dataset/HAM10000_images_part_1"
metadata_path = "D:/Dataset/HAM10000_metadata.csv"
img_size = (64, 64)

# Initialize lists to store image data and labels
images = []
labels = []

# Load metadata
metadata = pd.read_csv(metadata_path)

# Iterate over the image files
print("Loading and preprocessing images...")
for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    img_name = row['image_id'] + '.jpg'
    label = row['dx']
    
    img_path = os.path.join(image_folder_path, img_name)
    img = cv2.imread(img_path)
    
    if img is not None:
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        images.append(img)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    images, 
    labels_categorical,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# Define different model architectures for ensemble
def build_model_1(input_shape, num_classes):
    """Deeper CNN with residual-like connections"""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def build_model_2(input_shape, num_classes):
    """Wider CNN with more filters"""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def build_model_3(input_shape, num_classes):
    """CNN with smaller kernels and deeper architecture"""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Create data augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

# Train multiple models
input_shape = (64, 64, 3)
num_classes = labels_categorical.shape[1]
models = []
histories = []

# Build and train models
model_builders = [build_model_1, build_model_2, build_model_3]
for i, build_model in enumerate(model_builders):
    print(f"\nTraining Model {i+1}")
    model = build_model(input_shape, num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks
    )
    
    models.append(model)
    histories.append(history)

# Function for ensemble prediction
def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)

# Evaluate ensemble
print("\nEvaluating Ensemble Model")
ensemble_predictions = ensemble_predict(models, X_val)
ensemble_accuracy = np.mean(np.argmax(ensemble_predictions, axis=1) == np.argmax(y_val, axis=1))
print(f"Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")

# Save models
save_dir = "D:/Dataset/models"
os.makedirs(save_dir, exist_ok=True)

for i, model in enumerate(models):
    model_path = os.path.join(save_dir, f"skin_lesion_model_{i+1}.h5")
    try:
        model.save(model_path)
        print(f"Model {i+1} saved to {model_path}")
    except Exception as e:
        print(f"Error saving model {i+1}: {str(e)}")

import tensorflow as tf

# Load the model
model_path = 'D:/Dataset/models/skin_lesion_model_3.h5'
model = tf.keras.models.load_model(model_path)

# Plot training histories
plt.figure(figsize=(15, 5))

for i, history in enumerate(histories):
    plt.subplot(1, 3, i+1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model {i+1} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()

# Plot combined results
plt.figure(figsize=(10, 5))

# Plot final validation accuracies
individual_accuracies = [np.max(history.history['val_accuracy']) for history in histories]
accuracies = individual_accuracies + [ensemble_accuracy]
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), ['Model 1', 'Model 2', 'Model 3', 'Ensemble'])
plt.title('Model Comparison')
plt.ylabel('Validation Accuracy')
plt.show()

# Print final results
print("\nFinal Results:")
for i, acc in enumerate(individual_accuracies):
    print(f"Model {i+1} Best Validation Accuracy: {acc:.4f}")
print(f"Ensemble Model Validation Accuracy: {ensemble_accuracy:.4f}")
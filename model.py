import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
image_size = (128, 128)  # Image size for the model
batch_size = 32           # Batch size for training
train_split = 0.8        # 80% training, 20% validation

# Path to your dataset
dataset_path = 'dataset'  # Replace with the path to your dataset

# Count images in each class (Optional, for verification)
classes = ['HB', 'Normal', 'PMI']
for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    num_images = len(os.listdir(cls_path))
    print(f'Number of images in {cls}: {num_images}')

# Create ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255.0,               # Normalize pixel values to [0, 1]
    validation_split=1 - train_split,  # Split the data for validation
    horizontal_flip=True,              # Data augmentation: random horizontal flips
    rotation_range=20,                 # Random rotations
    width_shift_range=0.2,             # Random horizontal shifts
    height_shift_range=0.2             # Random vertical shifts
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',           # For multi-class classification
    subset='training',                   # Set as training data
    shuffle=True
)

# Load validation data
validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',           # For multi-class classification
    subset='validation',                 # Set as validation data
    shuffle=True
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes for HB, Normal, PMI
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10,  # Adjust epochs as needed
    steps_per_epoch=None,               # Set to None for automatic calculation
    validation_steps=None                # Set to None for automatic calculation
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_data)
print(f'Validation accuracy: {accuracy:.2f}')

# Save the model
model.save('cnn_model.h5')

# Plot training & validation accuracy and loss
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_training_history(history)

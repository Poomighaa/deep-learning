import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Function to predict class and display the image with disease recommendations
def predict_and_display_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Resize to match the model's expected input
    img_array = image.img_to_array(img)                      # Convert to array
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension
    img_array /= 255.0                                       # Normalize the image

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability

    # Class mapping
    class_labels = ['HB', 'Normal', 'PMI']
    disease_recommendations = {
        'HB': 'Recommendation: Consult a cardiologist for further evaluation.',
        'Normal': 'Recommendation: Continue regular health check-ups.',
        'PMI': 'Recommendation: Seek medical attention for potential issues.'
    }
    
    predicted_label = class_labels[predicted_class]
    recommendation = disease_recommendations[predicted_label]

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_label}\n{recommendation}')
    plt.axis('off')  # Hide axis
    plt.show()

# Example usage
image_path = '2.jpg'  # Replace with the path to your test image
predict_and_display_image(image_path)

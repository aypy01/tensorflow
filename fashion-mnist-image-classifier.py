# fashion_mnist_classifier.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Constants
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 1. Load & preprocess data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels

# 2. Build model
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

# 3. Train model
def train_model(model, train_images, train_labels):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)

# 4. Evaluate model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    color = '\033[92m' if test_accuracy > 0.90 else '\033[91m'
    print(f'{color}Test Accuracy: {test_accuracy:.3f}\033[0m')

# 5. Predict and plot
def plot_predictions(model, test_images, test_labels):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    def plot_image(i, predictions_array, true_label, img):
        predicted_label = np.argmax(predictions_array)
        color = 'blue' if predicted_label == true_label else 'red'
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):.0f}% ({class_names[true_label]})", color=color)

    def plot_value_array(i, predictions_array, true_label):
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        bars = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        bars[np.argmax(predictions_array)].set_color('red')
        bars[true_label].set_color('blue')

    # Display a grid of images with their predicted label and confidence bar chart

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols

    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels[i], test_images[i])

        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels[i])

    plt.tight_layout()
    plt.show()


# Entry point
if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    model = build_model()
    train_model(model, train_images, train_labels)
    evaluate_model(model, test_images, test_labels)
    plot_predictions(model, test_images, test_labels)

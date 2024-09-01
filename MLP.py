! pip install tensorflow matplotlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(activation):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Models with different activations
relu_model = build_model('relu')
leaky_relu_model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.01),
    Dense(64),
    LeakyReLU(alpha=0.01),
    Dense(10, activation='softmax')
])
leaky_relu_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

sigmoid_model = build_model('sigmoid')

# Train the models
history_relu = relu_model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
history_leaky_relu = leaky_relu_model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
history_sigmoid = sigmoid_model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)

print("ReLU Model Evaluation:")
relu_loss, relu_accuracy = relu_model.evaluate(X_test, y_test)
print(f"Test Loss: {relu_loss:.4f}, Test Accuracy: {relu_accuracy:.4f}\n")

print("LeakyReLU Model Evaluation:")
leaky_relu_loss, leaky_relu_accuracy = leaky_relu_model.evaluate(X_test, y_test)
print(f"Test Loss: {leaky_relu_loss:.4f}, Test Accuracy: {leaky_relu_accuracy:.4f}\n")

print("Sigmoid Model Evaluation:")
sigmoid_loss, sigmoid_accuracy = sigmoid_model.evaluate(X_test, y_test)
print(f"Test Loss: {sigmoid_loss:.4f}, Test Accuracy: {sigmoid_accuracy:.4f}\n")

def plot_history(histories, titles):
    plt.figure(figsize=(18, 6))  # Increase width for three plots
    for i, history in enumerate(histories):
        plt.subplot(1, 3, i+1)  # Change to 1 row and 3 columns
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(titles[i])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.show()

# Call the plot function with three histories and titles
plot_history([history_relu, history_leaky_relu, history_sigmoid], ['ReLU', 'LeakyReLU', 'Sigmoid'])



















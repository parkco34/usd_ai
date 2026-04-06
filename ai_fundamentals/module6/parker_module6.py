#!/usr/bin/env python
"""
Assignment 6.2 CNNs and Keras
-----------------------------
1. Load MNIST and rescale pixel intensities
2. Add channel dimension via Conv2D layers
3. Build CNN with two convolution-pool blocks using padding="same"
5. Plot learning curves and show 3 sample predictions
6. Provide reflection on three possible model improvements
"""
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Global constants
NUM_CLASSES = 10
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.10
STATE=73

def load_mnist():
    """
    Loads MNIST handwritten digits dataset for CNN
    ----------------------------------------------
    SHape of images are (28, 28), where we need to rescale the pixel values
    from (0, 255) to (0, 1) for the NNs training.
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Rescaling via min-max normalization 
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Channel aXis for grayscale images
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test

def sample_images(X_train, y_train, num=9):
    """
    Displays grid of MNIST training images, useful for quick sanity check,
    ensuring the images were loaded properly, labels match digits, etc.
    ---------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training images
        y_train: (np.ndarray) Training labels

    0UTPUT:
        None
    """
    plt.figure(figsize=(8, 8))

    # Iterate thru images for plot
    for thing in range(num):
        plt.subplot(3, 3, thing+1)
        # display dataset as 2D image
        plt.imshow(X_train[thing].squeez(), cmap="gray")
        plt.title(f"Label: {y_train[thing]}")
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Sample MNIST Training Images")
    plt.tight_layout()
    plt.show()

def build_cnn(input_shape=(28, 28, 1), class_num=10):
    """
    BUilds basic CNN
    ------------------------------------------
    INPUT:
        input_shape: (tuple) Shape of one input image
        class_num: (int) Number of output classes

    OUTPUT:
        model: (kera.Model) Uncompiled CNN model
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            # First convolution-pool block
            layers.Conv2D(
                filters=32, 
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # 2nd convolution-pool block
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Classifier head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(class_num, activation="softmax")
        ],
        name = "mnist_basic_cnn"
    )

    return model

def compile_model(model):
    """
    Compiles the CNN via optimizer and loss function for integer class labels
    ------------------------------------------------
    INPUT:
        model: (keras.model) CNN model

    OUTPUT:
        None
    """
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def training_model(model, X_train, y_train):
    """
    INPUT:
        model: (keras.model) CNN model
        x_train (np.ndarray): Training images
        y_train (np.ndarray): Training labels
        
    OUPUT:
        history: (keras.callback.History) Training history object w/ loss via epoch and accuracy values
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    return history

def model_eval(model, X_test, y_test):
    """
    Evaluates the CNN (duh)
    ----------------------------------
    INPUT:
        model: (keras.model) CNN model
        X_test: (np.ndarray) Test images
        y_test: (np.ndarray) Test labels
   
    OUTPUT:
        (tuple)
        test_loss: Final test loss
        test_accuracy: Final test accuracy
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(dedent(f"""\n
TEST PERFORMANCE
================
Test loss: {test_loss:.3f}
Test Accuracy: {test_accuracy:.3f}
                 """))

    return test_loss, test_accuracy

def plot_learning_curves(history):
    """
    PLots training and validation learning curves for both accuracy and loss.

    If training and validation accuracy rise together while loss decreases, the model is learning as it shud.
    If it improves quickly while validation gets worse, there might be overfitting.
    ------------------------------------------------
    INPUT:
        history: (keras.callbacks.History) History from fitting model

    OUTPUT:
        None
    """
    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Accuracy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_predictions(model, X_test, y_test, sample_indices=None):
    """
    Dispays sample predictions from test set.
    Shows:
        image
        predicted digit
        true digit
    ------------------------------------------------
    INPUT:
        model: (keras.model) CNN model
        X_test: (np.ndarray) Test images
        y_test: (np.ndarray) Test labels
        sample_indices: (None or int) Specific indices for display

    OUTPUT:
        None
    """
    # Cuz we don't want mutable objects in the function call, since python creates default arguments once @ function definition, not @ every call
    if sample_indices is None:
        sample_indices = [0, 1, 2]

    predictions = model.predict(X_test[sample_indices], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 3))

    for i, idx in enumerate(sample_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        plt.title(
            f"Predicted: {predicted_labels[i]}\nActual: {y_test[idx]}"
        )
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Sample CNN Predictions", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nThree sample predictions:")
    for i, idx in enumerate(sample_indices):
        print(
            f"Index {idx}: predicted = {predicted_labels[i]}, "
            f"actual = {y_test[idx]}"
        )

def interpret_convergence(history):
    """
    Outputs my interpretation of the model convergence
    """
    # Evaluations
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    
    # Ouputs
    print(dedent(f"""\n
The model seems to converge if training accuracy and validation accuracy both
                 increase over the 5 epochs while training and validation loss
                 decrease.

Final training accuracy: {final_train_acc:.3f}
Final validation accuuracy: {final_val_acc:.3f}
Final training loss: {final_train_loss:.3f}
Fina validation loss: {final_val_loss:3f}
                 """))
    # ?
    if abs(final_train_acc - final_val_acc) < 0.03:
        print(
            "Interpretation: the train/validation gap is fairly small, so "
            "the model does not show strong evidence of severe overfitting."
        )
    else:
        print(
            "Interpretation: there is a noticeable train/validation gap, "
            "which suggests some overfitting may be starting."
        )

def print_improvements():
    """
    Outputs model improvements w/ explanations.
    """
    print(dedent(f"""\n
3 MODEL IMRPOVEMENTS
--------------------
1) DROPTOUT REGULARIZATION
One improvement I would make is to add dropout after the dense layer or maybe
                 after the convolutional blocks.  
Dropout works by randomly turning off a fraction of neurons during training,
                 which prevents the network from depending too heavuly on any
                 one activation pattern.

2) BATCH NORMALIZATION
We could add bath normalization after the convolutional layers, where it
                 stabalizes the distribution of activations as they move
                 through the network, leading to faster and efficient training. 
It might help the optimizer converge more smoothly, where the training curves
                 would be less noisy.

3) DATA AUGMENTATION
We could use light data augmentation like small rotations, shifts, or zoom
                 operations.
Real handwritten digits can vary in small ways in position, angle, etc. So, by
                 exposing the CNN to these transformed versions during
                 traiining, the model would likely become more robust to
                 natural varations and improve its ability to classify new
                 digits.
                 """))

# ======= MAIN===========
np.random.seed(STATE)
tf.random.set_seed(STATE)

# Loading MNIST data
X_train, X_test, y_train, y_test = load_mnist()

# Output data info
print(dedent(f"""
\n
DATA INFO
X_train shape: {X_train.shape}
y_train shape: {y_train.shape}
X_test shape: {X_test.shape}
y_test shape: {y_test.shape}
             """))

# BUIlding CNN
model = build_cnn(
    input_shape=X_train.shape[1:],
    class_num=NUM_CLASSES
)

print(f"Model Summary:\n{model.summary()}")

# Compile/FIt CNN
compile_model(model)
history = training_model(model, X_train, y_train)

# Model evaluation
model_eval(model, X_test, y_test)
plot_learning_curves(history)
show_predictions(model, X_test, y_test, sample_indices=[0, 7, 19])
interpret_convergence(history)

# INterpretation/Improvements
print_improvements()











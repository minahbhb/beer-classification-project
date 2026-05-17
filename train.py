"""
Training pipeline for beer image classifier.
Author: Mina Habibi
"""

import os
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import data
import model


def plot_training_history(history, title="Training History"):
    """Plots training and validation accuracy and loss."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, eval_data, eval_labels):
    """Plots confusion matrix on evaluation data."""
    y_pred = model.predict(eval_data)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(eval_labels, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def train(params):
    """
    Loads data, trains the model, and evaluates performance.

    Parameters:
        params: argparse namespace with training parameters
    """
    print("Loading data...")
    train_data, train_labels = data.create_data_with_labels("data/train/")
    eval_data, eval_labels = data.create_data_with_labels("data/eval/")

    # Shuffle training data
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_gen = datagen.flow(
        train_data,
        train_labels,
        batch_size=model.get_batch_size()
    )

    # Build model
    img_shape = train_data.shape[1:]
    input_layer = tf.keras.Input(shape=img_shape, name='input_image')
    ml_model = model.solution(input_layer)

    # Early stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    print("Training model...")
    history = ml_model.fit(
        train_gen,
        epochs=model.get_epochs(),
        validation_data=(eval_data, eval_labels),
        callbacks=[callback],
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = ml_model.evaluate(eval_data, eval_labels, verbose=0)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss:     {val_loss:.4f}")

    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(ml_model, eval_data, eval_labels)

    # Save model
    ml_model.save("best_beer_model.keras")
    print("Model saved to best_beer_model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train beer image classifier"
    )
    args = parser.parse_args()

    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)

    train(args)
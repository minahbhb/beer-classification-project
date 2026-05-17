"""
Beer Classifier Model
MobileNetV2 with transfer learning and fine-tuning
Author: Mina Habibi
"""

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import MobileNetV2


def get_batch_size():
    """Returns the batch size for training."""
    return 16


def get_epochs():
    """Returns the maximum number of epochs for training."""
    return 150


def solution(input_layer):
    """
    Returns a compiled MobileNetV2 model with fine-tuning.

    Architecture:
        - MobileNetV2 backbone (ImageNet weights)
        - Bottom 150 layers frozen, top layers fine-tuned
        - Custom classification head with BatchNorm and Dropout
        - 5-class softmax output

    Parameters:
        input_layer: tf.keras.Input specifying image shape (H, W, 3)

    Returns:
        model: Compiled Keras model
    """
    # Load MobileNetV2 with ImageNet weights
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=input_layer,
        pooling="avg"
    )

    # Strategic fine-tuning: freeze bottom layers, unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:150]:
        layer.trainable = False

    # Custom classification head
    x = base_model.output
    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        512, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-2)
    )(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(5, activation="softmax")(x)

    model = models.Model(inputs=input_layer, outputs=output)

    # Learning rate schedule with exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.9
    )

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=lr_schedule,
        global_clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
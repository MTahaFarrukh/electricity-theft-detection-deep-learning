"""Keras model builders for the deep classifiers.

Input: (T, 2) per customer: channel 0 = preprocessed consumption,
channel 1 = missingness mask. T is padded to a whole number of weeks
(1035 days -> 1036 = 148 weeks) by `to_model_input`.

All builders return an UNCOMPILED model with a single sigmoid output (risk score).
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

DAYS_PER_WEEK = 7


def to_model_input(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Stack (values, mask) into (n, T', 2), padding T up to whole weeks with missing days."""
    pad = (-values.shape[1]) % DAYS_PER_WEEK
    values = np.pad(values, ((0, 0), (0, pad)))
    mask = np.pad(mask, ((0, 0), (0, pad)), constant_values=1)
    return np.stack([values, mask.astype(np.float32)], axis=-1).astype(np.float32)


def build_cnn(input_shape: tuple[int, int], dropout: float = 0.3) -> keras.Model:
    """1D-CNN: stacked convolutions learn local patterns (weekly cycles, sudden drops).

    The first kernel spans one week. Global average + max pooling summarise
    where (and how strongly) each learned pattern occurs over the whole series.
    """
    inputs = keras.Input(shape=input_shape, name="series")
    x = inputs
    for filters, kernel in [(32, 7), (64, 5), (64, 3)]:
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    return keras.Model(inputs, outputs, name="cnn")


def build_lstm(input_shape: tuple[int, int], dropout: float = 0.3) -> keras.Model:
    """Stand-alone LSTM over WEEKS: (T, 2) is reshaped to (T/7 weeks, 7 days x 2 channels).

    Reading 148 weekly steps instead of 1036 daily ones keeps training feasible
    on CPU and keeps the weekly structure the EDA found; the recurrent layers
    then model how consumption evolves week to week.
    """
    steps, channels = input_shape
    if steps % DAYS_PER_WEEK:
        raise ValueError(f"Sequence length {steps} is not a whole number of weeks; use to_model_input")
    inputs = keras.Input(shape=input_shape, name="series")
    x = layers.Reshape((steps // DAYS_PER_WEEK, DAYS_PER_WEEK * channels), name="to_weeks")(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    return keras.Model(inputs, outputs, name="lstm")


def build_cnn_lstm(input_shape: tuple[int, int], dropout: float = 0.3) -> keras.Model:
    """Hybrid CNN-LSTM (proposed model).

    The same convolutional blocks as the CNN extract local patterns and shorten
    the daily sequence 8x (1036 -> 129 steps of 64 learned features); the LSTM
    layers then model how those patterns evolve over the whole period,
    replacing the CNN's order-blind global pooling.
    """
    inputs = keras.Input(shape=input_shape, name="series")
    x = inputs
    for filters, kernel in [(32, 7), (64, 5), (64, 3)]:
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    return keras.Model(inputs, outputs, name="cnn_lstm")


BUILDERS = {"cnn": build_cnn, "lstm": build_lstm, "cnn_lstm": build_cnn_lstm}

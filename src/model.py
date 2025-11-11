"""
Defines a simple Siamese U-Net model for change detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_siamese_unet(input_shape=(256,256,3)):
    inp1 = layers.Input(shape=input_shape)
    inp2 = layers.Input(shape=input_shape)

    # Shared encoder
    concat_inputs = layers.Concatenate(axis=-1)([inp1, inp2])

    c1, p1 = encoder_block(concat_inputs, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)
    c4, p4 = encoder_block(p3, 256)

    b = conv_block(p4, 512)

    d1 = decoder_block(b, c4, 256)
    d2 = decoder_block(d1, c3, 128)
    d3 = decoder_block(d2, c2, 64)
    d4 = decoder_block(d3, c1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)

    model = Model(inputs=[inp1, inp2], outputs=outputs)
    return model

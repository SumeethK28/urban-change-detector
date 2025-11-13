"""
Defines a simple Siamese U-Net model for change detection.
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet_block(x, filters):
    c = Conv2D(filters, (3,3), activation='relu', padding='same')(x)
    c = Conv2D(filters, (3,3), activation='relu', padding='same')(c)
    return c

def SiameseUNet(input_shape=(256,256,3)):
    inpA = Input(shape=input_shape)
    inpB = Input(shape=input_shape)

    # Encoder shared weights
    def encoder(x):
        c1 = unet_block(x, 32)
        p1 = MaxPooling2D((2,2))(c1)
        c2 = unet_block(p1, 64)
        p2 = MaxPooling2D((2,2))(c2)
        c3 = unet_block(p2, 128)
        p3 = MaxPooling2D((2,2))(c3)
        return c1, c2, c3, p3

    c1A, c2A, c3A, p3A = encoder(inpA)
    c1B, c2B, c3B, p3B = encoder(inpB)

    diff = abs(p3A - p3B)

    u1 = UpSampling2D((2,2))(diff)
    u1 = concatenate([u1, abs(c3A - c3B)])
    u1 = unet_block(u1, 128)

    u2 = UpSampling2D((2,2))(u1)
    u2 = concatenate([u2, abs(c2A - c2B)])
    u2 = unet_block(u2, 64)

    u3 = UpSampling2D((2,2))(u2)
    u3 = concatenate([u3, abs(c1A - c1B)])
    u3 = unet_block(u3, 32)

    out = Conv2D(1, (1,1), activation='sigmoid')(u3)

    return Model(inputs=[inpA, inpB], outputs=out)

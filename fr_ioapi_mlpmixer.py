import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa



class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches



def build_classifier(blocks,in_shape,positional_encoding,embedding_dim,num_patches,num_classes,image_size,patch_size):
    
    dropout_rate = 0.2
    
    inputs = layers.Input(shape=in_shape)
    patches = Patches(patch_size, num_patches)(inputs)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embedding_dim
        )(positions)
        x = x + position_embedding
    # Process x using the module blocks.
    x = blocks(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    return keras.Model(inputs=inputs, outputs=logits)

class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate,embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=embedding_dim),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x
    


#%%

def ioapi_mlpmixer(in_shape=(80, 80, 3) , tune=0, classes=2):

    num_classes = classes
    embedding_dim = 256  # Number of hidden units.
    weight_decay = 0.0001
    dropout_rate = 0.2
    image_size = in_shape[0]  # We'll resize input images to this size.
    patch_size = 8  # Size of the patches to be extracted from the input images.
    num_patches = (image_size // patch_size) ** 2  # Size of the data array.
    num_blocks = 4  # Number of blocks.
    
    print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
    print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
    print(f"Patches per image: {num_patches}")
    print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
    
    
    mlpmixer_blocks = keras.Sequential(
        [MLPMixerLayer(num_patches, embedding_dim, dropout_rate,embedding_dim) for _ in range(num_blocks)]
    )
    
    learning_rate = 0.005
   
    mlpmixer_classifier = build_classifier(mlpmixer_blocks,in_shape,'False',embedding_dim,num_patches,num_classes,image_size,patch_size)
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    # Compile the model.
    mlpmixer_classifier.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc")
        ],
    )
    
    mlpmixer_classifier.summary()
    
    return mlpmixer_classifier


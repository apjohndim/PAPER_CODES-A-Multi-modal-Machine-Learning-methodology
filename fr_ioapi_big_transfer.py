from tensorflow import keras
import tensorflow_hub as hub


class MyBiTModel(keras.Model):
    def __init__(self, num_classes, module, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = module

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


def ioapi_big_transfer(in_shape=(80, 80, 3) , tune=0, classes=2):
    
    bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
    bit_module = hub.KerasLayer(bit_model_url)
    
    model = MyBiTModel(num_classes=classes, module=bit_module)
    optimizer = keras.optimizers.SGD(momentum=0.9)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    
    #model.summary()
    
    return model

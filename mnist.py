# step 1 import the modules
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Step 2 - loading the mnist data set 
(ds_train,ds_test), info=tfds.load('mnist',
                                split=['train','test'],
                                shuffle_files=True,
                                as_supervised=True,
                                with_info=True
)
# Step 3 - Build a trainig pipeline

# Initially the data is of type tf.unit8 but our model expects tf.float32......so we normalize it 
def Normalize(image,label):
    return tf.cast(image, tf.float32)/255.0, label

ds_train=ds_train.map(Normalize, num_parallel_calls=tf.data.AUTOTUNE)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(info.splits['train'].num_examples)
ds_train=ds_train.batch(128)
ds_train=ds_train.prefetch(tf.data.AUTOTUNE)

#step 4 - Building an evolution pipeline (basically same as above with the testing data set)

ds_test=ds_test.map(Normalize, num_parallel_calls=tf.data.AUTOTUNE)
ds_test=ds_test.cache()
ds_test=ds_test.batch(128)
ds_test=ds_test.prefetch(tf.data.AUTOTUNE)

#step 5 - create and train the model

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]

)
model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=10
)

model.save('model.keras')
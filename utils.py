import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import ds_test, ds_train
model=tf.keras.models.load_model('model.keras')

def predict_value(model, image_path):
    img=tf.keras.utils.load_img(image_path,target_size=(28,28),color_mode="grayscale")
    img_array=tf.keras.utils.img_to_array(img)
    img_array=img_array/ 255.0
    img_array=np.expand_dims(img_array,axis=0)
    pred=model.predict(img_array)
    result = np.argmax(pred,axis=1)
    return result

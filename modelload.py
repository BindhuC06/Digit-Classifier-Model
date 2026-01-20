import tensorflow as tf
from utils import predict_value

new_model=tf.keras.models.load_model('model.keras')

result=predict_value(new_model, './3.jpg')
print("The predicted number is : ",result)

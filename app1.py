import tensorflow as tf
import pandas as pd


model = tf.keras.models.load_model('birdvdrone.h5')
data = pd.read_csv('C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\BE_Project\\Datasets\\birdcsv\\bird3.csv')
predictions = model.predict(data)
print(predictions)

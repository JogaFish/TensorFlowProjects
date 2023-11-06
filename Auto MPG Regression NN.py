import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
tf.get_logger().setLevel("ERROR")

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
features = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
mean_horsepower = features['Horsepower'].mean()
features['Horsepower'].fillna(mean_horsepower, inplace=True)
NUMERIC_COLUMNS = ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
                   'Acceleration', 'Model Year']
CATEGORICAL_COLUMNS = ['Origin']

y = features.pop('MPG')
x_train = features.sample(frac=0.8, random_state=0)
x_test = features.drop(x_train.index)
y_train = y.sample(frac=0.8, random_state=0)
y_test = y.drop(y_train.index)


def build_model():
    dnn_model = Sequential([
        Dense(units=64, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])
    dnn_model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
    return dnn_model


model = build_model()
model.fit(x_train, y_train, epochs=100, batch_size=32)
y_pred = model.predict(x_test).flatten()
error = y_pred - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()
print(error.mean())
print(error.median())





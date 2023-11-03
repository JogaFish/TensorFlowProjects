from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

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

for col in NUMERIC_COLUMNS:
    features[col] = (features[col] - features[col].mean())/features[col].std()
features['MPG'] = (features['MPG'] - features['MPG'].mean())/features['MPG'].std()

labels = features.pop('MPG')
train_features = features.sample(frac=0.8, random_state=0)
test_features = features.drop(train_features.index)
train_labels = labels.sample(frac=0.8, random_state=0)
test_labels = labels.drop(train_labels.index)


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = train_features[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_fn(data_df, label_df, num_epochs=500, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use


# here we will call the input_function that was returned to us to get a dataset object we can feed to the model
for i in range(60):
    train_input_fn = make_input_fn(train_features, train_labels)

eval_input_fn = make_input_fn(test_features, test_labels, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)
print(result)



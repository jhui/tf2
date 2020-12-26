# Modified from
# https://www.tensorflow.org/tutorials/load_data/csv

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

inputs = {}

for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# concatenate the numeric inputs together, and run them through a normalization layer:
numeric_inputs = {name: input for name, input in inputs.items()
                  if input.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

# Create preprocessed inputs for the model
preprocessed_inputs = [all_numeric_inputs]

# convert category type into one-hot vector
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

# Concatenate all input
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

# Create a pre-processing model
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

titanic_features_dict = {name: np.array(value)
                         for name, value in titanic_features.items()}

features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)


def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam())
    return model


titanic_model = titanic_model(titanic_preprocessing, inputs)

titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}

result = titanic_model(features_dict)
print(result)


# Modified from
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

# !pip install -q git+https://github.com/tensorflow/docs
import tensorflow as tf

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import pathlib
import shutil
import tempfile

# Remove TensorBoard log directory
logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# Higgs Dataset contains 28 features
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28

# Use tf.data to create a dataset
ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type="GZIP")


# Repack data as (feature_vectors, labels)
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


# Perform the mapping in batch for faster performance
# Un-batch the ds afterward
packed_ds = ds.batch(10000).map(pack_row).unbatch()

# Just for demonstration,
# use a smaller subset of data for training
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

# Partition the training dataset into training & validation
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# Create a custom schedule for learning rate decay
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)


# Create Adam optimizer with explicit learning rate
def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),  # Just print a single dot for each epoch
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        # Also enable TensorBoard logging during training
        tf.keras.callbacks.TensorBoard(logdir / name),
    ]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'),
                      'accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history


combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories = {}
regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

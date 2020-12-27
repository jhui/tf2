# Modified from
# https://www.tensorflow.org/tutorials/images/data_augmentation

import tensorflow as tf
import tensorflow_datasets as tfds

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

image, label = next(iter(train_ds))

flipped = tf.image.flip_left_right(image)
grayscaled = tf.image.rgb_to_grayscale(image)
saturated = tf.image.adjust_saturation(image, 3)
bright = tf.image.adjust_brightness(image, 0.4)
cropped = tf.image.central_crop(image, central_fraction=0.5)
rotated = tf.image.rot90(image)

IMG_SIZE = 180

def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label


def augment(image, label):
    image, label = resize_and_rescale(image, label)
    # Add 6 pixels of padding
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    image = tf.clip_by_value(image, 0, 1)
    return image, label


batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = (
    train_ds
        .shuffle(1000)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
        .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
        .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
)

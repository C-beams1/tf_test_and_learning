import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pathlib  # pathlib is in standard library

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# img_height = 28
# img_width = 28
# batch_size = 2
#
# model = keras.Sequential([
#     layers.Input((28, 28, 1)),
#     layers.Conv2D(16, 3, padding='same'),
#     layers.Conv2D(32, 3, padding='same'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(10),
# ])
#
# # method 1 Using dataset_from_directory
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     'data/mnist_subfolders/',
#     labels='inferred',
#     label_mode='int',  # categorical, binary
#     color_mode='grayscale',
#     batch_size=batch_size,
#     img_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,  # training in same mode every time
#     validation_split=0.1,
#     subset='training',
# )
#
# ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
#     'data/mnist_subfolders/',
#     labels='inferred',
#     label_mode='int',  # categorical, binary
#     color_mode='grayscale',
#     batch_size=batch_size,
#     img_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,  # training in same mode every time
#     validation_split=0.1,
#     subset='validation',
# )
#
#
# def augment(x, y):
#     image = tf.image.random_brightness(x, max_delta=0.05)
#     return image, y
#
#
# ds_train = ds_train.map(augment)
#
# # Custom Loops
# for epochs in range(10):
#     for x, y in ds_train:
#         # train here
#         pass
#
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[
#         keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     ],
#     metrics=['accuracy'],
# )
#
# model.fit(ds_train, epochs=10, verbose=2)
#
#
# # method 2 ImageDataGenerator and flow_from_directory
# datagen = ImageDataGenerator(
#     rescale=1./255,  # make sure that it's in float
#     rotation_range=5,
#     zoom_range=(0.95, 0.95),
#     horizontal_flip=False,
#     vertical_flip=False,
#     data_format='channels_last',  # standard way
#     validation_split=0.0,
#     dtype=tf.float32,
# )
#
# train_generator = datagen.flow_from_directory(
#     'data/mnist_subfolders/',
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     color_mode='grayscale',
#     class_mode='sparse',  # we want int rather than one hot encodings
#     shuffle=True,  # randomize
#     subset='training',
#     seed=123,
# )
#
#
# def training(): pass
#
#
# # Custom Loops
# for epoch in range(10):
#     num_batchs = 0
#
#     for x, y in ds_train:
#         num_batchs += 1
#
#         # do training
#         training()
#
#         if num_batchs == 25:  # len(train_dataset)/batch_size
#             break
#
#
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[
#         keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     ],
#     metrics=['accuracy'],
# )
#
# model.fit(train_generator, epochs=10, steps_per_epoch=25, verbose=2)

# directory = 'data/mnist_images_csv/'
# df = pd.read_csv(directory + 'train.csv')
#
# file_paths = df['file_name'].values
# labels = df['label'].values
# ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
#
#
# def read_image(image_file, label):
#     image = tf.io.read_file(directory + image_file)
#     image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
#     return image, label
#
#
# def augment(image, label):
#     return image, label
#
#
# ds_train = ds_train.map(read_image).map(augment).batch(2)
#
# for epoch in range(10):
#     for x, y in ds_train:
#         # train here
#         pass
#
# model = keras.Sequential([
#     layers.Input((28, 28, 1)),
#     layers.Conv2D(16, 3, padding='same'),
#     layers.Conv2D(32, 3, padding='same'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(10),
# ])
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[
#         keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     ],
#     metrics=['accuracy'],
# )
#
# model.fit(ds_train, epochs=10, verbose=2)

batch_size = 2
img_heigtht = 28
img_width = 28

directory = 'data/mnist_images_only'
ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.jpg')))


def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)
    label = tf.strings.split(file_path, '\\')
    label = tf.strings.substr(label, pos=0, len=1)[2]
    label = tf.strings.to_number(label, out_type=tf.int64)
    return image, label


ds_train = ds_train.map(process_path).batch(batch_size)

model = keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics=['accuracy'],
)

model.fit(ds_train, epochs=10, verbose=2)
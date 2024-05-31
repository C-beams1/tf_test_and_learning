import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import pickle


def filter_train(line):
    split_line = tf.strings.split(line, ',', maxsplit=4)
    dataset_belonging = split_line[1]  # train, test
    sentiment_catagory = split_line[2]  # pos neg unsupervised

    return (
        True
        if dataset_belonging == 'train' and sentiment_catagory != 'unsup'
        else False
    )


def filter_test(line):
    split_line = tf.strings.split(line, ',', maxsplit=4)
    dataset_belonging = split_line[1]  # train, test
    sentiment_catagory = split_line[2]  # pos neg unsupervised

    return (
        True
        if dataset_belonging == 'test' and sentiment_catagory != 'unsup'
        else False
    )


ds_train = tf.data.TextLineDataset('imdb.csv').filter(filter_train)
ds_test = tf.data.TextLineDataset('imdb.csv').filter(filter_train)

# for line in ds_trian.skip(1).take(5):
#     print(tf.strings.split(line, ',', maxsplit=4))
# TODO:
# 1. Create vocabulary
# 2. Numericalize text str -> indices (TokenTextEncoder)
# 3. Pad the batches so we can send in to an RNN for example


# 'i love banana' -> ['i', 'love', 'banana'] -> [0, 1 ,2]

# file_names = ['test_example1.csv', 'test_example2.csv', 'test_example3.csv']
# dataset = tf.data.TextLineDataset(file_names)

# dataset1 = tf.data.TextLineDataset('text_example1.csv').skip(1)#.map(preprocessing)
# dataset2 = tf.data.TextLineDataset('text_example2.csv').skip(1)#.map(preprocessing)
# dataset3 = tf.data.TextLineDataset('text_example3.csv').skip(1)#.map(preprocessing)
# dataset = dataset1.concatenate(dataset2).concatenate(dataset3)

english = tf.data.TextLineDataset('english.csv')
chinese = tf.data.TextLineDataset('chinese.csv')
dataset = tf.data.Dataset.zip((english, chinese))

for eng, chn in dataset.skip(1):
    print(tokenizer.tokenize(eng.numpy()))
    print(tokenizer.tokenize(chn.numpy().decode('UTF-8')))


def build_vocabulary(ds_train, threshold=200):
    frequencies = {}
    vocabulary = set()
    vocabulary.update(['sostoken'])  # start token
    vocabulary.update(['eostoken'])  # end token

    for line in ds_train.skip(1):
        split_line = tf.strings.split(line, ',', maxsplit=4)
        review = split_line[4]
        tokenized_text = tokenizer.tokenize(review.numpy().lower())

    for word in tokenized_text:
        if word not in frequencies:
            frequencies[word] = 1
        else:
            frequencies[word] += 1

        if frequencies[word] == threshold:
            vocabulary.update(tokenized_text)

    return vocabulary


# Build, save, load
vocabulary = build_vocabulary(ds_train)
vocab_file = open('vocabulary.obj', 'wb')
pickle.dump(vocabulary, vocab_file)
# vocab_file = open('vocabulary.obj', 'rb')
# vocabulary = pickle.load(vocab_file)

tokenizer = tfds.deprecated.text.Tokenizer
encoder = tfds.deprecated.text.TokenTextEncoder(
    list(vocabulary), oov_token='<UNK>', lowercase=True, tokenizer=tokenizer
)


def my_encoder(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(line):
    split_line = tf.strings.split(line, ',', maxsplit=4)
    label_str = split_line[2]  # neg, pos
    review = 'sostoken' + split_line[4] + 'eostoken'
    label = 1 if label_str == 'pos' else 0

    (encoded_text, label) = tf.py_function(
        my_encoder, inp=[review, label], Tout=(tf.int64, tf.int32)
    )

    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map_fn, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(5000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))

ds_test = ds_test.map(encode_map_fn)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

model = keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocabulary)+2, output_dim=32),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ]
)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(3e-4, clipnorm=1),
    metrics=['accuracy'],
)

model.fit(ds_train, epochs=15, verbose=2)
model.evaluate(ds_test)
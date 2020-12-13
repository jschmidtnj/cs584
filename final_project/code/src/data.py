#!/usr/bin/env python3
"""
data file

read in data
"""

import pandas as pd
import tensorflow as tf
from loguru import logger
from utils import file_path_relative, roc_auc
from variables import raw_data_folder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel, TFRobertaPreTrainedModel
from tokenizers import BertWordPieceTokenizer


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)

    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def read_data(strategy) -> pd.DataFrame:
    """
    read data from raw data, convert to dataframes
    """
    logger.info('TODO - read data')
    train = pd.read_csv(file_path_relative(
        f'{raw_data_folder}/jigsaw-toxic-comment-train.csv'))
    validation = pd.read_csv(file_path_relative(
        f'{raw_data_folder}/validation.csv'))
    test = pd.read_csv(file_path_relative(f'{raw_data_folder}/test.csv'))

    train.drop(['severe_toxic', 'obscene', 'threat', 'insult',
                'identity_hate'], axis=1, inplace=True)
    train = train.loc[:12000, :]
    logger.info(train.shape)

    max_len = train['comment_text'].apply(
        lambda x: len(str(x).split())).max()
    logger.info(f'max len: {max_len}')

    xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values,
                                                      stratify=train.toxic.values,
                                                      test_size=0.2, shuffle=True)

    token = tf.keras.preprocessing.text.Tokenizer(num_words=None)

    token.fit_on_texts(list(xtrain) + list(xvalid))
    xtrain_seq = token.texts_to_sequences(xtrain)
    xvalid_seq = token.texts_to_sequences(xvalid)

    # zero pad the sequences
    xtrain_pad = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain_seq, maxlen=max_len)
    xvalid_pad = tf.keras.preprocessing.sequence.pad_sequences(
        xvalid_seq, maxlen=max_len)

    word_index = token.word_index

    # SimpleRNN

    with strategy.scope():
        # A simpleRNN without any pretrained embeddings and one dense layer
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(len(word_index) + 1,
                                            300,
                                            input_length=max_len))
        model.add(tf.keras.layers.SimpleRNN(100))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    # model.fit(xtrain_pad, ytrain, batch_size=64*strategy.num_replicas_in_sync)

    # scores = model.predict(xvalid_pad)
    # logger.info(f"AUC: {roc_auc(scores, yvalid):.2f}")

    embeddings_index = {}
    with open(file_path_relative(f'{raw_data_folder}/glove.840B.300d.txt'), encoding='utf-8') as glove_file:
        for line in tqdm(glove_file):
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray([float(val) for val in values[1:]])
            embeddings_index[word] = coefs

    logger.info(f'Found {len(embeddings_index)} word vectors.')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # LSTM

    with strategy.scope():
        # A simple LSTM with glove embeddings and one dense layer
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(len(word_index) + 1,
                                            300,
                                            weights=[embedding_matrix],
                                            input_length=max_len,
                                            trainable=False))

        model.add(tf.keras.layers.LSTM(
            100, dropout=0.3, recurrent_dropout=0.3))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(xtrain_pad, ytrain, nb_epoch=5,
              batch_size=64*strategy.num_replicas_in_sync)

    scores = model.predict(xvalid_pad)
    logger.info(f"AUC: {roc_auc(scores, yvalid):.2f}")

    # GRU

    with strategy.scope():
        # GRU with glove embeddings and two dense layers
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(len(word_index) + 1,
                                            300,
                                            weights=[embedding_matrix],
                                            input_length=max_len,
                                            trainable=False))
        model.add(tf.keras.layers.SpatialDropout1D(0.3))
        model.add(tf.keras.layers.GRU(300))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(xtrain_pad, ytrain, nb_epoch=5,
              batch_size=64*strategy.num_replicas_in_sync)

    scores = model.predict(xvalid_pad)
    logger.info(f"AUC: {roc_auc(scores, yvalid):.2f}")

    # bidirectional RNN

    with strategy.scope():
        # A simple bidirectional LSTM with glove embeddings and one dense layer
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(len(word_index) + 1,
                                            300,
                                            weights=[embedding_matrix],
                                            input_length=max_len,
                                            trainable=False))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(xtrain_pad, ytrain, nb_epoch=5,
              batch_size=64*strategy.num_replicas_in_sync)

    scores = model.predict(xvalid_pad)
    logger.info(f"AUC: {roc_auc(scores, yvalid):.2f}")

    # Attention models

    # IMP DATA FOR CONFIG

    AUTO = tf.data.experimental.AUTOTUNE

    # Configuration
    EPOCHS = 3
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    MAX_LEN = 192

    # First load the real tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-multilingual-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

    train1 = pd.read_csv(file_path_relative(
        f'{raw_data_folder}/jigsaw-toxic-comment-train.csv'))
    valid = pd.read_csv(file_path_relative(
        f'{raw_data_folder}/validation.csv'))

    x_train = fast_encode(train1.comment_text.astype(str),
                          fast_tokenizer, maxlen=MAX_LEN)
    x_valid = fast_encode(valid.comment_text.astype(str),
                          fast_tokenizer, maxlen=MAX_LEN)
    x_test = fast_encode(test.content.astype(
        str), fast_tokenizer, maxlen=MAX_LEN)

    y_train = train1.toxic.values
    y_valid = valid.toxic.values

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )

    with strategy.scope():
        transformer_layer = (
            TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased')
        )
        model = build_model(transformer_layer, max_len=MAX_LEN)
    model.summary()

    n_steps = x_train.shape[0] // BATCH_SIZE
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS
    )

    n_steps = x_valid.shape[0] // BATCH_SIZE
    train_history_2 = model.fit(
        valid_dataset.repeat(),
        steps_per_epoch=n_steps,
        epochs=EPOCHS*2
    )

    scores = model.predict(test_dataset, verbose=1)
    logger.info(f"AUC: {roc_auc(scores, yvalid):.2f}")

    # Roberta

    with strategy.scope():
        transformer_layer = TFRobertaPreTrainedModel()
        model = build_model(transformer_layer, max_len=MAX_LEN)
    model.summary()

    n_steps = x_train.shape[0] // BATCH_SIZE
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS
    )

    n_steps = x_valid.shape[0] // BATCH_SIZE
    train_history_2 = model.fit(
        valid_dataset.repeat(),
        steps_per_epoch=n_steps,
        epochs=EPOCHS
    )

    scores = model.predict(test_dataset, verbose=1)
    logger.info(f"AUC: {roc_auc(scores, yvalid):.2f}")


if __name__ == '__main__':
    # data = read_data()
    # logger.info(data.head())
    pass

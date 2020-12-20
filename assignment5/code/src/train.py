#!/usr/bin/env python3
"""
train file

run training on dataset
"""

import tensorflow as tf
from loguru import logger
from os.path import join
import time
from typing import List
import matplotlib.pyplot as plt
from utils import file_path_relative
from variables import output_folder, IN_NOTEBOOK


def _loss_function(real, pred, loss_object):
    """
    loss function for training
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def _train_step(inp, targ, enc_hidden, targ_lang, encoder, decoder, loss_object, optimizer, BATCH_SIZE):
    """
    training step in model
    """
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims(
            [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)

            loss += _loss_function(targ[:, t], predictions, loss_object)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def _plot_train_val_loss(training_loss: List[float], model_name: str) -> None:
    """
    plots the training and validation loss given history
    """

    plt.figure()

    num_epochs = len(training_loss)
    nums = range(1, num_epochs + 1)

    plt.plot(nums, training_loss, label="train")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if IN_NOTEBOOK:
        plt.show()
    else:
        file_path = file_path_relative(
            f'{output_folder}/{model_name}.jpg')
        plt.savefig(file_path)


def run_train(input_tensor_train, target_tensor_train, targ_lang, checkpoint,
              checkpoint_dir, encoder, optimizer, decoder, steps_per_epoch,
              BUFFER_SIZE, BATCH_SIZE, EPOCHS, model_name: str) -> None:
    """
    create and run training
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    checkpoint_prefix = join(checkpoint_dir, "ckpt")

    training_loss: List[float] = []
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = _train_step(inp, targ, enc_hidden, targ_lang,
                                     encoder, decoder, loss_object, optimizer, BATCH_SIZE)
            total_loss += batch_loss
            if batch % 100 == 0:
                logger.info(
                    f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        epoch_loss = total_loss / steps_per_epoch
        training_loss.append(epoch_loss)
        logger.info(f'Epoch {epoch + 1} Loss {epoch_loss:.4f}')
        logger.info(f'Time taken for epoch {time.time() - start} sec\n')

    _plot_train_val_loss(training_loss, model_name)

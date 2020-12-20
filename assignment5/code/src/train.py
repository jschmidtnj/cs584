#!/usr/bin/env python3
"""
lstm file

run lstm on dataset
"""

import tensorflow as tf
from loguru import logger
from os.path import join
import time


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


def run_train(input_tensor_train, target_tensor_train, targ_lang, checkpoint, checkpoint_dir, encoder, optimizer, decoder, steps_per_epoch, BUFFER_SIZE, BATCH_SIZE) -> None:
    """
    create and run training
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    checkpoint_prefix = join(checkpoint_dir, "ckpt")

    EPOCHS = 15

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

        logger.info(
            f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        logger.info(f'Time taken for epoch {time.time() - start} sec\n')


if __name__ == '__main__':
    raise RuntimeError('cannot run training on its own')

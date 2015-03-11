import glob
import time
import os

import numpy as np


def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx


def get_val_error_loss(rand_arr, shared_x, shared_y,
                       val_filenames, val_labels,
                       flag_datalayer,
                       batch_size, validate_model,
                       send_queue=None, recv_queue=None):

    if flag_datalayer:
        rand_arr.set_value(np.float32([0.5, 0.5, 0]))

    validation_losses = []
    validation_errors = []

    n_val_batches = len(val_filenames)

    for val_index in range(n_val_batches):
        val_img = np.load(str(val_filenames[val_index]))
        shared_x.set_value(val_img)

        shared_y.set_value(val_labels[val_index * batch_size:
                                      (val_index + 1) * batch_size])
        loss, error = validate_model()

        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)

    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)

    return this_validation_error, this_validation_loss


def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand


def train_model_wrap(train_model, shared_x, shared_y, rand_arr, img_mean,
                     count, minibatch_index, minibatch_range, batch_size,
                     train_filenames, train_labels, flag_datalayer):

    batch_img = np.load(str(train_filenames[minibatch_index])) - img_mean
    shared_x.set_value(batch_img)

    batch_label = train_labels[minibatch_index * batch_size:
                               (minibatch_index + 1) * batch_size]
    shared_y.set_value(batch_label)

    if flag_datalayer:
        rand_arr.set_value(get_rand3d())

    cost_ij = train_model()

    return cost_ij

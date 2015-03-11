import sys
import time
import glob
import numpy as np

from alexnet.tools import (save_weights, load_weights,
                           save_momentums, load_momentums)
from alexnet.train_funcs import (adjust_learning_rate,
                                 get_val_error_loss, get_rand3d,
                                 train_model_wrap)


def unpack_configs(config):
    flag_datalayer = config['use_data_layer']

    # Load Training/Validation Filenames and Labels
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    label_folder = config['label_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*.npy'))
    val_filenames = sorted(glob.glob(val_folder + '/*.npy'))
    train_labels = np.load(label_folder + 'train_labels.npy')
    val_labels = np.load(label_folder + 'val_labels.npy')
    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    return (flag_datalayer, train_filenames, val_filenames,
            train_labels, val_labels, img_mean)


def train_net(config):
    # UNPACK CONFIGS
    (flag_datalayer, train_filenames, val_filenames,
     train_labels, val_labels, img_mean) = unpack_configs(config)

    import theano
    theano.config.on_unused_input = 'warn'

    from alexnet.layers import DropoutLayer
    from alexnet.alex_net import AlexNet, compile_models

    ## BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size

    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
        shared_x, shared_y, rand_arr, vels) = compile_models(model, config)


    ######################### TRAIN MODEL ################################

    print '... training'

    n_train_batches = len(train_filenames)
    minibatch_range = range(n_train_batches)

    # Start Training Loop
    epoch = 0
    step_idx = 0
    val_record = []
    while epoch < config['n_epochs']:
        epoch = epoch + 1

        if config['shuffle']:
            np.random.shuffle(minibatch_range)

        if config['resume_train'] and epoch == 1:
            load_epoch = config['load_epoch']
            load_weights(layers, config['weights_dir'], load_epoch)
            epoch = load_epoch + 1
            lr_to_load = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            val_record = list(
                np.load(config['weights_dir'] + 'val_record.npy'))
            learning_rate.set_value(lr_to_load)
            load_momentums(vels, config['weights_dir'], epoch)

        count = 0
        for minibatch_index in minibatch_range:

            num_iter = (epoch - 1) * n_train_batches + count
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)

            cost_ij = train_model_wrap(train_model, shared_x,
                                       shared_y, rand_arr, img_mean,
                                       count, minibatch_index,
                                       minibatch_range, batch_size,
                                       train_filenames, train_labels,
                                       flag_datalayer)


            if num_iter % config['print_freq'] == 0:
                print 'training @ iter = ', num_iter
                print 'training cost:', cost_ij
                if config['print_train_error']:
                    print 'training error rate:', train_error()

        ############### Test on Validation Set ##################

        DropoutLayer.SetDropoutOff()

        this_validation_error, this_validation_loss = get_val_error_loss(
            rand_arr, shared_x, shared_y, val_filenames, val_labels,
            flag_datalayer, batch_size, validate_model)

        print('epoch %i: validation loss %f ' %
              (epoch, this_validation_loss))
        print('epoch %i: validation error %f %%' %
              (epoch, this_validation_error * 100.))
        val_record.append([this_validation_error, this_validation_loss])
        np.save(config['weights_dir'] + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()
        ############################################

        # Adapt Learning Rate
        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                        val_record, learning_rate)

        # Save weights
        if epoch % config['snapshot_freq'] == 0:
            save_weights(layers, config['weights_dir'], epoch)
            np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate.get_value())
            save_momentums(vels, config['weights_dir'], epoch)

    print('Optimization complete.')

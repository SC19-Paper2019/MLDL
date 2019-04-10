#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Modified from https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py 
and https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_tune_cifar10_with_keras.py
"""

from __future__ import print_function

import argparse

import numpy as np
import os 
from tensorflow.keras.utils import multi_gpu_model 
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import ray
from ray.tune import grid_search, run_experiments, sample_from
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining

num_classes = 10

def load_data():
  """Loads CIFAR10 dataset.
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  from tensorflow.python.keras.datasets.cifar import load_batch
  from tensorflow.python.keras import backend as K
  dirname = 'cifar-10-batches-py'
  #origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  #path = get_file(dirname, origin=origin, untar=True)
  path = '/mnt/bb/%s/%s'%(os.environ['USER'],dirname)
  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  #if K.image_data_format() == 'channels_last':
  #x_train = x_train.transpose(0, 2, 3, 1)
  #x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)


class Cifar10Model(Trainable):
    def _read_data(self):
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = load_data()
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train = x_train.astype("float32")
        x_train /= 255
        x_test = x_test.astype("float32")
        x_test /= 255

        return (x_train, y_train), (x_test, y_test)
    def _resnet_layer(self, inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    batch_normalization=True,
                    conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
               bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
               x = BatchNormalization()(x)
            if activation is not None:
               x = Activation(activation)(x)
        else:
            if batch_normalization:
               x = BatchNormalization()(x)
            if activation is not None:
               x = Activation(activation)(x)
            x = conv(x)
        return x


    def _resnet_v1(self, input_shape, depth, num_classes=10):
        """ResNet Version 1 Model builder [a]
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self._resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
               strides = 1
               if stack > 0 and res_block == 0:  # first layer but not first stack
                   strides = 2  # downsample
               y = self._resnet_layer(inputs=x,
                           num_filters=num_filters,
                           strides=strides)
               y = self._resnet_layer(inputs=y,
                           num_filters=num_filters,
                           activation=None)
               if stack > 0 and res_block == 0:  # first layer but not first stack
                   # linear projection residual shortcut connection to match
                   # changed dims
                   x = self._resnet_layer(inputs=x,
                               num_filters=num_filters,
                               kernel_size=1,
                               strides=strides,
                               activation=None,
                               batch_normalization=False)
               x = tf.keras.layers.add([x, y])
               x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                     activation='softmax',
                     kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _build_model(self, input_shape, depth):
        print(input_shape, depth)
        model = self._resnet_v1(input_shape=input_shape, depth=depth)
        return model

    def _setup(self, config):
        self.train_data, self.test_data = self._read_data()
        x_train = self.train_data[0]
        model = self._build_model(input_shape=x_train.shape[1:], depth=self.config["depth"])
        parallel_model = multi_gpu_model(model, gpus=6)
        opt = tf.keras.optimizers.Adam(
            lr=self.config["lr"], decay=self.config["decay"])
        parallel_model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])
        self.model = parallel_model

    def _train(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data

        aug_gen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by dataset std
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
        )

        aug_gen.fit(x_train)
        gen = aug_gen.flow(
            x_train, y_train, batch_size=self.config["batch_size"])
        self.model.fit_generator(
            generator=gen,
            steps_per_epoch=50000 // self.config["batch_size"],
            epochs=self.config["epochs"],
            validation_data=None)

        # loss, accuracy
        _, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {"mean_accuracy": accuracy}

    def _save(self, checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, "model.h5")
        self.model.save_weights(file_path)
        return file_path

    def _restore(self, path):
        self.model.load_weights(path)

    def _stop(self):
        # If need, save your model when exit.
        # saved_path = self.model.save(self.logdir)
        # print("save model at: ", saved_path)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        '--redis-address', default=None, type=str, help='Finish quickly for testing')
    parser.add_argument(
        '--output-dir', default=None, type=str, help='Finish quickly for testing')
    args, _ = parser.parse_known_args()

    train_spec = {
        "run": Cifar10Model,
        "resources_per_trial": {
            "cpu": 42,
            "gpu": 6
        },
        "stop": {
            "mean_accuracy": 0.90,
            "training_iteration": 50,
        },
        "config": {
            "epochs": 1,
            "batch_size": 64*6,
            "lr": grid_search([10**-3, 10**-4]),
            "decay": sample_from(lambda spec: spec.config.lr / 10.0),
            "depth": grid_search([20,32,44,50]),
        },
        "local_dir": args.output_dir,
        "num_samples": 8,
      #  "checkpoint_freq":1,
    }

    if args.smoke_test:
        train_spec["config"]["lr"] = 10**-4
        train_spec["config"]["depth"] = 20

    ray.init(redis_address=args.redis_address, log_to_driver=False)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="mean_accuracy",
        perturbation_interval=10,
        hyperparam_mutations={
            "decay": [10**-4, 10**-5, 10**-6],
        })

    run_experiments({"pbt_cifar10": train_spec}, scheduler=pbt)

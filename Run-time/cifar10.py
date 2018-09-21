
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_ops

from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import serial

from collections import OrderedDict

if __name__ == "__main__":

    batch_size = 4000
    print("batch_size = "+str(batch_size))

    convOp = "myconv"
    #convOp = "cublas"
    #convOp = "cudnn"
    print("convOp = "+ convOp)

    #for convOp cublas, cudnn, set kernel=theano
    kernel = "myxnor"
    # kernel = "theano"
    print("kernel = "+ kernel)

    print('Loading CIFAR10 dataset...')

    test_set = CIFAR10(which_set= 'test')
    # Inputs in the range [-1,+1]
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    # flatten targets
    test_set.y = test_set.y.reshape(-1)
    test_set.X = test_set.X[:batch_size]
    test_set.y = test_set.y[:batch_size]

    print('Building the CovNet...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.vector('targets')

    cnn = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input)

    cnn = binary_ops.Conv2DLayer(
            cnn,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity,
            convOp="cudnn")

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.Conv2DLayer(
            cnn,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity,
            convOp=convOp)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.Conv2DLayer(
            cnn,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity,
            convOp=convOp)

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.Conv2DLayer(
            cnn,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity,
            convOp=convOp)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity,
            convOp=convOp)

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.Conv2DLayer(
            cnn,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity,
            convOp=convOp)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024,
            kernel=kernel)

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024,
            kernel=kernel)

    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_ops.SignTheano)

    cnn = binary_ops.DenseLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10,
            kernel=kernel)

    cnn = lasagne.layers.BatchNormLayer(cnn)


    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), target),dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], test_err)

    print("Loading the trained parameters and binarizing the weights...")

    # Load parameters
    #with np.load('cifar10_parameters_test.npz') as f:
    with np.load('cifar10_parameters.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)

    # Binarize the weights
    params = lasagne.layers.get_all_params(cnn)
    for param in params:
        # print param.name
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))

    print('Running...')
    print("convOp, batch_size, test_error, time, imgs")

    iters = 10
    seqw = [1, 10, 50, 100, 200, 500, 1000]
    for i in seqw:
        ter = 0.0
        rt = 0.0
        for j in range(1, 11):
            start_time = time.time()
            test_error = val_fn(test_set.X[:i],test_set.y[:i])*100.
            run_time = time.time() - start_time
            ter = ter + test_error
            rt = rt + run_time

        ter = ter/iters
        rt = rt/iters
        ims = i/rt
        print(convOp + ", " + str(i) + ", " + str(ter) + ", " +str(rt)+ ", " + str(ims))


import tensorflow as tf


def get_optimizer(config):
    if config.train.optimizer == 'adam':
        optimizer = tf.optimizers.Adam(lr=config.train.lr)

    elif config.train.optimizer == 'sgd':
        optimizer = tf.optimizers.SGD(lr=config.train.lr, momentum=0.9)

    else:
        raise NotImplementedError

    return optimizer

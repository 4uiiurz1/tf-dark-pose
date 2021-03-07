import os
import time
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from omegaconf import OmegaConf
from loguru import logger

from src.datasets import get_dataset
from src.models import get_model
from src.losses import JointsMSELoss
from src.optimizers import get_optimizer
from src.metrics import accuracy
from src.postprocess import get_final_preds


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('config_path',
                        default='configs/res50_128x96_d256x3_adam_lr1e-3.yaml',
                        help='Path to config file.')

    args = parser.parse_args()

    return args


def get_lr(config, epoch):
    lr = config.train.lr
    factor = config.train.lr_factor
    steps = config.train.lr_step

    for step in steps:
        if epoch >= step:
            lr *= factor

    return lr


@tf.function
def train_on_batch(model, criterion, optimizer, data):
    with tf.GradientTape() as tape:
        outputs = model(data['inputs'])
        loss = criterion(outputs, data['target'], data['target_weight'])
        grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return outputs, loss


def train_on_epoch(config, model, criterion, optimizer, train_dataset, train_generator):
    losses = tf.keras.metrics.Mean()
    accs = tf.keras.metrics.Mean()
    data_time = tf.keras.metrics.Mean()
    batch_time = tf.keras.metrics.Mean()

    pbar = tqdm(total=int(np.ceil(len(train_dataset) / config.train.batch_size)))

    end = time.time()
    for step, data in enumerate(train_generator):
        # Measure data loading time
        data_time(time.time() - end)

        # Compute output and gradient and update weights
        outputs, loss = train_on_batch(model, criterion, optimizer, data)
        losses(loss)

        # Calculate accuracy according to PCK
        _, acc, _, _ = accuracy(
            outputs.numpy().transpose(0, 3, 1, 2),
            data['target'].numpy().transpose(0, 3, 1, 2),
        )
        accs(acc)

        # Measure elapsed time
        batch_time(time.time() - end)
        end = time.time()

        pbar.set_postfix_str('loss: %e, acc: %f, data_time: %f, batch_time: %f' % (
            losses.result(),
            accs.result(),
            data_time.result(),
            batch_time.result(),
        ))
        pbar.update(1)

    pbar.close()

    metrics = {
        'train_loss': losses.result(),
        'train_acc': accs.result(),
        'train_data_time': data_time.result(),
        'train_batch_time': batch_time.result(),
    }

    for name, metric in metrics.items():
        logger.info('%s: %f' % (name, metric))

    return metrics


@tf.function
def val_on_batch(model, criterion, data):
    outputs = model(data['inputs'])
    loss = criterion(outputs, data['target'], data['target_weight'])

    return outputs, loss


def val_on_epoch(config, model, criterion, val_dataset, val_generator, output_dir):
    losses = tf.keras.metrics.Mean()
    accs = tf.keras.metrics.Mean()
    data_time = tf.keras.metrics.Mean()
    batch_time = tf.keras.metrics.Mean()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.model.num_joints, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    img_path = []
    idx = 0

    pbar = tqdm(total=int(np.ceil(len(val_dataset) / config.test.batch_size)))

    end = time.time()
    for step, data in enumerate(val_generator):
        # Measure data loading time
        data_time(time.time() - end)

        # Compute output
        outputs, loss = val_on_batch(model, criterion, data)
        losses(loss)

        # Calculate accuracy according to PCK
        _, acc, _, _ = accuracy(
            outputs.numpy().transpose(0, 3, 1, 2),
            data['target'].numpy().transpose(0, 3, 1, 2),
        )
        accs(acc)

        # Measure elapsed time
        batch_time(time.time() - end)
        end = time.time()

        c = data['center'].numpy()
        s = data['scale'].numpy()
        score = data['score'].numpy()
        num_images = len(data['inputs'])

        preds, maxvals = get_final_preds(
            config,outputs.numpy().transpose(0, 3, 1, 2), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        img_path.extend(data['image_file'].numpy().tolist())

        idx += num_images

        pbar.set_postfix_str('loss: %e, acc: %f, data_time: %f, batch_time: %f' % (
            losses.result(),
            accs.result(),
            data_time.result(),
            batch_time.result(),
        ))
        pbar.update(1)

    pbar.close()

    coco_metrics, _ = val_dataset.evaluate(
        all_preds, output_dir, all_boxes, img_path)

    metrics = {
        'val_loss': losses.result(),
        'val_acc': accs.result(),
        'val_data_time': data_time.result(),
        'val_batch_time': batch_time.result(),
    }
    metrics.update(coco_metrics)

    for name, metric in metrics.items():
        logger.info('%s: %f' % (name, metric))

    return metrics


def init_gpu():
    # https://qiita.com/masudam/items/c229e3c75763e823eed5
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    args = _parse_args()

    # Get config
    with open(args.config_path, 'r') as f:
        config = OmegaConf.load(f)

    # Create model directory
    model_dir = os.path.join('models', config.name)
    os.makedirs(model_dir, exist_ok=True)

    # Setup logger
    logger.add(os.path.join(model_dir, 'train.log'))

    # Initialize GPU
    init_gpu()

    # Model
    model = get_model(config)

    # Loss
    criterion = JointsMSELoss(use_target_weight=config.loss.use_target_weight)

    # Optimizer
    optimizer = get_optimizer(config)

    # Checkpoint
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(model_dir, 'checkpoints'), max_to_keep=5)

    # Restore checkpoint
    if checkpoint_manager.latest_checkpoint:
        logger.info('Restored checkpoint from %s.' % checkpoint_manager.latest_checkpoint)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        initial_epoch = checkpoint.epoch.numpy()
    else:
        initial_epoch = 0

    # Dataset
    train_dataset = get_dataset(
        config,
        image_set=config.dataset.train_set,
        is_train=True,
    )
    train_generator = tf.data.Dataset.from_generator(
        train_dataset.generator,
        train_dataset.output_types,
    )
    train_generator = train_generator.map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_generator = train_generator.batch(config.train.batch_size, drop_remainder=True)
    train_generator = train_generator.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = get_dataset(
        config,
        image_set=config.dataset.test_set,
        is_train=False,
    )
    val_generator = tf.data.Dataset.from_generator(
        val_dataset.generator,
        val_dataset.output_types,
    )
    val_generator = val_generator.map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_generator = val_generator.batch(config.test.batch_size)
    val_generator = val_generator.prefetch(tf.data.experimental.AUTOTUNE)

    summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'logs'))

    best_ap = 0

    for epoch in range(initial_epoch, config.train.num_epochs):
        logger.info('Epoch: %d/%d' % (epoch + 1, config.train.num_epochs))

        optimizer.lr = get_lr(config, epoch)

        train_metrics = train_on_epoch(config, model, criterion, optimizer, train_dataset, train_generator)
        val_metrics = val_on_epoch(config, model, criterion, val_dataset, val_generator, model_dir)

        with summary_writer.as_default():
            for name, metric in train_metrics.items():
                tf.summary.scalar(name, metric, step=epoch)
            for name, metric in val_metrics.items():
                tf.summary.scalar(name, metric, step=epoch)

        # Save checkpoint
        checkpoint.epoch.assign_add(1)
        checkpoint_manager.save()

        if val_metrics['ap'] > best_ap:
            best_ap = val_metrics['ap']
            model.save(os.path.join(model_dir, 'best_model.hdf5'))


if __name__ == '__main__':
    main()

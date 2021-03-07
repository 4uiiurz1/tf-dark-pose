import random

import pytest
from easydict import EasyDict as edict
import yaml
import tensorflow as tf

from src.datasets import get_dataset


def test_coco_shuffle():
    random.seed(71)

    with open('data/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = edict(config)

    dataset = get_dataset(
        config,
        image_set=config.dataset.test_set,
        is_train=True,
    )

    generator = tf.data.Dataset.from_generator(
        dataset.generator,
        dataset.output_types,
    )
    generator = generator.map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    generator = generator.batch(config.train.batch_size_per_gpu)
    generator = generator.prefetch(tf.data.experimental.AUTOTUNE)

    file_names_list = []
    for epoch in range(2):
        file_names = []
        for data in generator:
            file_names.extend(data['file_name'].numpy().tolist())
        file_names_list.append(file_names)

    assert len(file_names_list[0]) == len(file_names_list[1])

    matched_cnt = 0
    for file_name1, file_name2 in zip(file_names_list[0], file_names_list[1]):
        if file_name1 == file_name2:
            matched_cnt += 1

    assert matched_cnt / len(file_names_list[0]) <= 0.005

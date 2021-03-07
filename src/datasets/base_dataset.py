"""
Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
Original licence: Copyright (c) Microsoft, under the MIT License.
"""

from abc import ABC, abstractmethod
import copy
import random

import cv2
import numpy as np
from albumentations import (
    Compose,
    Normalize,
)
import tensorflow as tf

from ..transforms import (
    get_affine_transform,
    affine_transform,
    fliplr_joints,
    half_body_transform,
)


class BaseDataset(ABC):
    def __init__(self, config, root: str, image_set: str, is_train: str):
        self.num_joints = 0

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.scale_factor = config.dataset.scale_factor
        self.rotation_factor = config.dataset.rot_factor
        self.flip = config.dataset.flip
        self.half_body_prob = config.dataset.half_body_prob
        self.half_body_num_joints = config.dataset.half_body_num_joints
        self.color_rgb = config.dataset.color_rgb

        self.input_w = config.model.input_width
        self.input_h = config.model.input_height
        self.output_w = config.model.output_width
        self.output_h = config.model.output_height
        self.aspect_ratio = self.input_w * 1.0 / self.input_h

        self.sigma = config.model.sigma

        self.transform = Compose([
            Normalize(
                mean=config.model.mean,
                std=config.model.std),
        ])

        self.flip_pairs = None
        self.upper_body_ids = None
        self.lower_body_ids = None

        self.db = []

        self.output_types = {
            'inputs': tf.float32,
            'target': tf.float32,
            'target_weight': tf.float32,
            'image_file': tf.string,
            'file_name': tf.string,
            'joints': tf.float32,
            'joints_visible': tf.float32,
            'center': tf.float32,
            'scale': tf.float32,
            'rotation': tf.float32,
            'score': tf.float32,
        }

    @abstractmethod
    def _get_db(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, preds, output_dir, all_boxes, img_path):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        file_name = db_rec['file_name'] if 'file_name' in db_rec else ''

        img = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))

        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_3d']
        joints_visible = db_rec['joints_3d_visible']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        # augmentation
        if self.is_train:
            if (np.sum(joints_visible[:, 0]) > self.half_body_num_joints
                    and np.random.rand() < self.half_body_prob):
                c_half_body, s_half_body = half_body_transform(
                    joints,
                    joints_visible,
                    self.num_joints,
                    self.upper_body_ids,
                    self.aspect_ratio)

                if c_half_body is not None and s_half_body is not None:
                    c = c_half_body
                    s = s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                img = img[:, ::-1, :]
                joints, joints_visible = fliplr_joints(
                    joints, joints_visible, img.shape[1], self.flip_pairs)
                c[0] = img.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, (self.input_w, self.input_h))
        img = cv2.warpAffine(
            img,
            trans,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            img = self.transform(image=img)['image']

        for i in range(self.num_joints):
            if joints_visible[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self._generate_target(joints, joints_visible)

        return {
            'inputs': img,
            'target': target,
            'target_weight': target_weight,
            'image_file': image_file,
            'file_name': file_name,
            'joints': joints,
            'joints_visible': joints_visible,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

    def generator(self):
        """Generator for `tf.data.Dataset.from_generator`.
        """
        indices = list(range(len(self)))
        if self.is_train:
            random.shuffle(indices)
        for idx in indices:
            yield self[idx]

    def _generate_target(self, joints: np.ndarray, joints_visible: np.ndarray):
        """Generate target map.

        Args:
            joints : [num_joints, 3]
            joints_visible: [num_joints, 3]

        Returns:
            target: [self.output_height, self.output_width, num_joints]
            target_weight: 1: visible, 0: invisible

        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_visible[:, 0]

        target = np.zeros(
            (self.output_h, self.output_w, self.num_joints),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            heatmap_vis = joints_visible[joint_id, 0]
            target_weight[joint_id] = heatmap_vis

            feat_stride = [
                self.input_w / self.output_w,
                self.input_h / self.output_h,
            ]
            mu_x = joints[joint_id][0] / feat_stride[0]
            mu_y = joints[joint_id][1] / feat_stride[1]
            # Check that any part of the gaussian is in-bounds
            ul = [mu_x - tmp_size, mu_y - tmp_size]
            br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
            if (ul[0] >= self.output_w or
                    ul[1] >= self.output_h or
                    br[0] < 0 or
                    br[1] < 0):
                target_weight[joint_id] = 0

            if target_weight[joint_id] == 0:
                continue

            x = np.arange(0, self.output_w, 1, np.float32)
            y = np.arange(0, self.output_h, 1, np.float32)
            y = y[:, None]

            if target_weight[joint_id] > 0.5:
                target[..., joint_id] = np.exp(
                    -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        return target, target_weight

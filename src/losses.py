import tensorflow as tf


class JointsMSELoss:
    def __init__(self, use_target_weight):
        self.criterion = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.use_target_weight = use_target_weight

    def __call__(self, output, target, target_weight):
        batch_size, _, _, num_joints = output.shape

        heatmaps_pred = tf.transpose(output, perm=[0, 3, 1, 2])
        heatmaps_pred = tf.reshape(heatmaps_pred, shape=[batch_size, num_joints, -1])
        heatmaps_pred = tf.split(heatmaps_pred, num_or_size_splits=num_joints, axis=1)

        heatmaps_gt = tf.transpose(target, perm=[0, 3, 1, 2])
        heatmaps_gt = tf.reshape(heatmaps_gt, shape=[batch_size, num_joints, -1])
        heatmaps_gt = tf.split(heatmaps_gt, num_or_size_splits=num_joints, axis=1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = tf.squeeze(heatmaps_pred[idx])
            heatmap_gt = tf.squeeze(heatmaps_gt[idx])
            if self.use_target_weight:
                loss += 0.5 * tf.reduce_mean(self.criterion(
                    heatmap_gt * target_weight[:, idx],
                    heatmap_pred * target_weight[:, idx],
                ))
            else:
                loss += 0.5 * tf.reduce_mean(self.criterion(heatmap_gt, heatmap_pred))

        return loss / num_joints


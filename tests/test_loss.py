import pytest
import numpy as np

from src.losses import JointsMSELoss


def test_joint_mse_loss():
    """Check if loss matches pytorch.

    Expected value is created by the following code:

    .. code-block:: python
        import torch.nn as nn

        class JointsMSELoss(nn.Module):
            def __init__(self, use_target_weight):
                super().__init__()
                self.criterion = nn.MSELoss(reduction='mean')
                self.use_target_weight = use_target_weight

            def forward(self, output, target, target_weight):
                batch_size = output.size(0)
                num_joints = output.size(1)
                heatmaps_pred = output.reshape(
                    (batch_size, num_joints, -1)).split(1, 1)
                heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
                loss = 0

                for idx in range(num_joints):
                    heatmap_pred = heatmaps_pred[idx].squeeze()
                    heatmap_gt = heatmaps_gt[idx].squeeze()
                    if self.use_target_weight:
                        loss += 0.5 * self.criterion(
                            heatmap_pred.mul(target_weight[:, idx]),
                            heatmap_gt.mul(target_weight[:, idx])
                        )
                    else:
                        loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

                return loss / num_joints


        if __name__ == '__main__':
            import numpy as np
            import torch

            np.random.seed(71)

            output = torch.from_numpy(np.random.rand(32, 32, 24, 17).astype(np.float32))
            output = output.permute(0, 3, 1, 2)
            target = torch.from_numpy(np.random.rand(
                32, 32, 24, 17).astype(np.float32))
            target = target.permute(0, 3, 1, 2)
            target_weight = torch.from_numpy(
                np.random.rand(32, 17, 1).astype(np.float32))

            criterion = JointsMSELoss(use_target_weight=True)
            loss = criterion(output, target, target_weight)

            print(loss.numpy())
    """
    np.random.seed(71)

    output = np.random.rand(32, 32, 24, 17).astype(np.float32)
    target = np.random.rand(32, 32, 24, 17).astype(np.float32)
    target_weight = np.random.rand(32, 17, 1).astype(np.float32)

    criterion = JointsMSELoss(use_target_weight=True)

    loss = criterion(output, target, target_weight)
    loss = loss.numpy()

    assert loss == pytest.approx(0.030641317, abs=1e-9)

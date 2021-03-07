import logging

import tensorflow as tf


logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.9   # 1 - 0.1


class LayerList(tf.keras.layers.Layer):
    def __init__(self, layers: list):
        super().__init__()
        self.layers = layers

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def conv1x1(out_planes, stride=1):
    return tf.keras.layers.Conv2D(
        filters=out_planes,
        kernel_size=1,
        strides=stride,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
    )


def conv3x3(out_planes, stride=1):
    return LayerList([
        tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
        tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            strides=stride,
            padding='valid',
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        ),
    ])


def bn():
    return tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = bn()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = conv3x3(planes)
        self.bn2 = bn()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlock(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(planes)
        self.bn1 = bn()
        self.conv2 = conv3x3(planes, stride=stride)
        self.bn2 = bn()
        self.conv3 = conv1x1(planes * self.expansion)
        self.bn3 = bn()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(tf.keras.layers.Layer):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = tf.keras.layers.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = LayerList([
                conv1x1(
                    num_channels[branch_index] * block.expansion,
                    stride=stride,
                ),
                bn(),
            ])

        layers = [block(
            num_channels[branch_index],
            stride,
            downsample
        )]
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index]
                )
            )

        return LayerList(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        LayerList([
                            conv1x1(num_inchannels[i]),
                            bn(),
                            tf.keras.layers.UpSampling2D(size=2**(j - i), interpolation='nearest'),
                        ])
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.extend([
                                conv3x3(num_outchannels_conv3x3, stride=2),
                                bn(),
                            ])
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.extend([
                                conv3x3(num_outchannels_conv3x3, stride=2),
                                bn(),
                                tf.keras.layers.ReLU(),
                            ])
                    fuse_layer.append(LayerList(conv3x3s))
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def call(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': BottleneckBlock,
}


class HighResolutionNet(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.model.extra
        super().__init__()

        # stem net
        self.conv1 = conv3x3(64, stride=2)
        self.bn1 = bn()
        self.conv2 = conv3x3(64, stride=2)
        self.bn2 = bn()
        self.relu = tf.keras.layers.ReLU()
        self.layer1 = self._make_layer(BottleneckBlock, 64, 4)

        self.stage2_cfg = cfg.model.extra.stage2
        num_channels = self.stage2_cfg.num_channels
        block = blocks_dict[self.stage2_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg.model.extra.stage3
        num_channels = self.stage3_cfg.num_channels
        block = blocks_dict[self.stage3_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg.model.extra.stage4
        num_channels = self.stage4_cfg.num_channels
        block = blocks_dict[self.stage4_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = tf.keras.layers.Conv2D(
            filters=cfg.model.num_joints,
            kernel_size=extra.final_conv_kernel,
            strides=1,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        )

        self.pretrained_layers = cfg.model.extra.pretrained_layers

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        LayerList([
                            conv3x3(num_channels_cur_layer[i]),
                            bn(),
                            tf.keras.layers.ReLU(),
                        ])
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.extend([
                        conv3x3(outchannels, stride=2),
                        bn(),
                        tf.keras.layers.ReLU(),
                    ])
                transition_layers.append(LayerList(conv3x3s))

        return transition_layers

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LayerList([
                conv1x1(planes * block.expansion, stride=stride),
                bn(),
            ])

        layers = [block(planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes))

        return LayerList(layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = blocks_dict[layer_config.block]
        fuse_method = layer_config.fuse_method

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return LayerList(modules), num_inchannels

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.num_branches):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    # def init_weights(self, pretrained=''):
    #
    #     if os.path.isfile(pretrained):
    #         pretrained_state_dict = torch.load(pretrained)
    #         logger.info('=> loading pretrained model {}'.format(pretrained))
    #
    #         need_init_state_dict = {}
    #         for name, m in pretrained_state_dict.items():
    #             if name.split('.')[0] in self.pretrained_layers \
    #                or self.pretrained_layers[0] is '*':
    #                 need_init_state_dict[name] = m
    #         self.load_state_dict(need_init_state_dict, strict=False)
    #     elif pretrained:
    #         logger.error('=> please download pre-trained models first!')
    #         raise ValueError('{} is not exist!'.format(pretrained))


def get_hrnet(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)

    return model


if __name__ == '__main__':
    import numpy as np
    import yaml
    from easydict import EasyDict as edict

    # model = BasicBlock(32)
    # model = BottleneckBlock(32)
    # model = HighResolutionModule()
    with open('../../configs/w32_128x96_adam_lr1e-3.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)

    model = HighResolutionNet(config)
    model.build(input_shape=(1, 96, 128, 3))
    model.summary()

    inputs = np.zeros((1, 96, 128, 3), dtype=np.float32)
    outputs = model(inputs)
    print(outputs)
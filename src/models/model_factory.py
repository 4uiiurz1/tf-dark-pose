from .resnet import get_resnet


def get_model(config):
    if config.model.name.startswith('resnet'):
        model = get_resnet(
            name=config.model.name,
            input_width=config.model.input_width,
            input_height=config.model.input_height,
            num_joints=config.model.num_joints,
            final_conv_kernel=config.model.extra.final_conv_kernel,
            num_deconv_filters=config.model.extra.num_deconv_filters,
            num_deconv_kernels=config.model.extra.num_deconv_kernels,
            deconv_with_bias=config.model.extra.deconv_with_bias,
        )
    else:
        raise ValueError

    return model

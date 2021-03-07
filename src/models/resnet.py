import tensorflow as tf


def _get_backbone(name: str, input_width: int, input_height: int):
    if name == 'resnet50':
        backbone_builder = tf.keras.applications.ResNet50V2
    elif name == 'resnet101':
        backbone_builder = tf.keras.applications.ResNet101V2
    elif name == 'resnet152':
        backbone_builder = tf.keras.applications.ResNet152V2
    else:
        raise ValueError

    backbone = backbone_builder(
        include_top=False,
        input_shape=(input_height, input_width, 3),
    )

    return backbone


def get_resnet(name: str,
               input_width: int,
               input_height: int,
               num_joints: int,
               final_conv_kernel: int,
               num_deconv_filters: list,
               num_deconv_kernels: list,
               deconv_with_bias: bool):
    inputs = tf.keras.layers.Input(shape=(input_height, input_width, 3), name='input')

    # Backbone
    backbone = _get_backbone(name, input_width, input_height)
    backbone = tf.keras.Model(inputs=backbone.inputs, outputs=backbone.outputs)
    x = backbone(inputs)

    # Deconv layers
    for num_deconv_filter, num_deconv_kernel in zip(num_deconv_filters,
                                                    num_deconv_kernels):
        x = tf.keras.layers.Conv2DTranspose(
            filters=num_deconv_filter,
            kernel_size=num_deconv_kernel,
            strides=2,
            padding='same',
            use_bias=deconv_with_bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Head
    outputs = tf.keras.layers.Conv2D(
        filters=num_joints,
        kernel_size=final_conv_kernel,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model

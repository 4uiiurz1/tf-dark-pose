from .coco import COCODataset


def get_dataset(config, **kwargs):
    if config.dataset.name == 'coco':
        dataset = COCODataset(
            config,
            root=config.dataset.root,
            image_set=kwargs['image_set'],
            is_train=kwargs['is_train'],
        )

    else:
        raise NotImplementedError

    return dataset

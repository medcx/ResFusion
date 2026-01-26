from basic.data.fusion_dataset import FusionImageDataset, DemoDataset


def create_dataset(dataset_config):
    if dataset_config['type'] == 'fusion':
        dataset = FusionImageDataset(**dataset_config['params'])
    elif dataset_config['type'] == 'demo':
        dataset = DemoDataset(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])
    return dataset

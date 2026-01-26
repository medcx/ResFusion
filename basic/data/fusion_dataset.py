import torch
from torch.utils.data import Dataset
import os
import cv2
from models.warped import warp2D, ImageTransform_1
import numpy as np
from glob import glob


class FusionImageDataset(Dataset):
    def __init__(self, dataset_path, source_modal, target_modal, warp_modal, isTrain, test=False):
        if isTrain:
            self.data_path = os.path.join(dataset_path, 'train')
        else:
            self.data_path = os.path.join(dataset_path, 'test')
        self.test = test
        self.train = isTrain
        self.input_modal = source_modal
        self.output_modal = target_modal
        self.warp_modal = warp_modal
        self.data_set = os.listdir(os.path.join(self.data_path, source_modal))
        self.n_data = len(self.data_set)
        self.warper = ImageTransform_1()

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        name = self.data_set[item]
        data_list = []
        data_CrCb = 0

        for modality in [self.input_modal, self.output_modal, self.warp_modal]:
            if modality == self.warp_modal:
                if self.train:
                    modality = self.input_modal
                    warped = True
                else:
                    warped = False
            else:
                warped = False
            data_name = os.path.join(self.data_path, modality, name)
            if ('CT' in modality or 'MRI' in modality or 'T1' in modality or 'T2' in modality or 't1' in modality
                    or 't2' in modality) and modality != 'SPECT':
                data = cv2.imread(data_name, cv2.IMREAD_GRAYSCALE)
                data = (data / 255.0 * 2 - 1).astype(np.float32)
                data = torch.from_numpy(data[None])
            else:
                data = cv2.imread(data_name)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2YCrCb)
                data_CrCb = data[:, :, 1:3].transpose(2, 0, 1)
                data_CrCb = data_CrCb / 255.0
                data = data[:, :, 0:1]
                data = data.squeeze()
                data = (data / 255.0 * 2 - 1).astype(np.float32)
                data = torch.from_numpy(data[None])
            if warped:
                # data, flow = self.warper(data[None] * 0.5 + 0.5)
                # data = data[0] * 2 - 1
                _, flow = self.warper(data[None] * 0.5 + 0.5)
                flow = flow.permute(0, 3, 1, 2) * 256
                data = warp2D()(data[None], flow).squeeze(0)
            data_list.append(data)
        if self.test:
            return {'A': data_list[0], 'B': data_list[1], 'A_warp': data_list[2],
                    'patient': name, 'CrCb': data_CrCb}

        return {'A': data_list[0], 'B': data_list[1], 'A_warp': data_list[2]}


class DemoDataset(Dataset):
    def __init__(self, dataset_path, isTrain, test=False):
        self.data_path = dataset_path
        self.test = test
        self.train = isTrain
        self.data_set = glob(f'{dataset_path}/B*.png')
        self.n_data = len(self.data_set)
        self.warper = ImageTransform_1()

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        name = self.data_set[item]
        data_CrCb = 0
        data_B = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        data_A = cv2.imread(name.replace('B', 'A'), cv2.IMREAD_GRAYSCALE)
        data_A_warp = cv2.imread(name.replace('B_modality', 'warped_A'), cv2.IMREAD_GRAYSCALE)
        data_B = (data_B / 255.0 * 2 - 1).astype(np.float32)
        data_A = (data_A / 255.0 * 2 - 1).astype(np.float32)
        data_A_warp = (data_A_warp / 255.0 * 2 - 1).astype(np.float32)
        data_B = torch.from_numpy(data_B[None])
        data_A = torch.from_numpy(data_A[None])
        data_A_warp = torch.from_numpy(data_A_warp[None])

        return {'A': data_A, 'B': data_B, 'A_warp': data_A_warp,
                'patient': name, 'CrCb': data_CrCb}

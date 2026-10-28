import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import json
from .build import DATASETS
from utils.logger import *


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

        # Stage-2.5: load PCN proxy difficulty JSON if present (pre-computed by
        # tools/compute_pcn_proxy_difficulty.py). This is a best-effort load at
        # the dataset layer: absence here is NOT fatal, but the runner WILL
        # fail-hard downstream for UpTrans_DiffCond on PCN when the proxy is
        # missing (runner enforces the real experimental invariant). Here we
        # just populate self.difficulty_map so __getitem__ can look up per-sample
        # labels; -1 sentinel returned when the sample key is not in the map.
        # See the supplementary PCN sanity-check section.
        # Proxy path is configurable via DIFFICULTY_PROXY_PATH in the dataset
        # config (alternate PCN roots or non-repo working directories).
        self.difficulty_map = {}
        self.proxy_path = getattr(config, 'DIFFICULTY_PROXY_PATH',
                                  'data/PCN/difficulty_proxy.json')
        try:
            if os.path.isfile(self.proxy_path):
                with open(self.proxy_path, 'r') as f:
                    proxy = json.loads(f.read())
                subset_map = proxy.get('difficulty', {}).get(self.subset, {})
                if isinstance(subset_map, dict) and subset_map:
                    self.difficulty_map = subset_map
                    print_log(
                        f'PCN proxy difficulty loaded ({len(self.difficulty_map)} '
                        f'entries for subset={self.subset}) from {self.proxy_path}',
                        logger='PCNDATASET')
                else:
                    print_log(
                        f'PCN proxy file {self.proxy_path} has no entries for '
                        f'subset={self.subset}', logger='PCNDATASET')
            else:
                print_log(
                    f'PCN proxy difficulty NOT present at {self.proxy_path} '
                    f'(subset={self.subset}); __getitem__ returns -1 sentinel.',
                    logger='PCNDATASET')
        except Exception as e:  # noqa: BLE001 - best-effort, never block training
            print_log(
                f'PCN proxy difficulty load failed (ignored): {e}',
                logger='PCNDATASET')

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        # Stage-2.5: look up proxy difficulty if available.
        # Sentinel -1 = "no proxy / do not apply DiffCond conditioning on this sample".
        # Key format matches tools/compute_pcn_proxy_difficulty.py output:
        #   train: "<taxonomy_id>/<model_id>/<rand_idx:02d>"
        #   test:  "<taxonomy_id>/<model_id>"
        diff_label = -1
        if self.difficulty_map:
            if self.subset == 'train':
                key = f"{sample['taxonomy_id']}/{sample['model_id']}/{rand_idx:02d}"
            else:
                key = f"{sample['taxonomy_id']}/{sample['model_id']}"
            diff_label = int(self.difficulty_map.get(key, -1))

        # Inner tuple stays backward-compatible for callers that only read
        # data[0] (partial) and data[1] (gt); data[2] is the new diff_label.
        return sample['taxonomy_id'], sample['model_id'], \
               (data['partial'], data['gt'], diff_label)

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class PCNv2(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)
    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
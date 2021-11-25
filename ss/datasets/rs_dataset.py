# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import paddle
import numpy as np
from .raster import Raster
from ss.transforms import Compose


class RSDataset(paddle.io.Dataset):
    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes,
                 mode='train',
                 file_path=None,
                 separator=' ',
                 ignore_index=255,
                 rgb_bands=[1, 2, 3],
                 big_map=False, 
                 grid_size=[512, 512],
                 overlap=[0, 0]):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, rgb_bands)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if mode.lower() not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if file_path is None:
            raise ValueError(
                '`file_path` is necessary, but it is None.'
            )
        elif not os.path.exists(file_path):
            raise FileNotFoundError(
                '`file_path` is not found: {}'.format(file_path))
        else:
            file_path = file_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                label_worker = None if label_path is None else \
                               Raster(label_path, [1, 1, 1], big_map, grid_size, overlap)
                self.file_list.append([
                    Raster(image_path, rgb_bands, big_map, grid_size, overlap), 
                    label_worker])

    def __getitem__(self, idx):
        image_worker, label_worker = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_worker.getData())
            im = im[np.newaxis, ...]
            return im, image_worker.file_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=image_worker.getData())
            label = label_worker.getData()
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_worker.getData(), 
                                        label=label_worker.getData())
            return im, label

    def __len__(self):
        return len(self.file_list)
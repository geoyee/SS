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
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
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

        if mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError(
                    '`train_path` is not found: {}'.format(train_path))
            else:
                file_path = train_path
        elif mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))
            else:
                file_path = test_path

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
                self.file_list.append([Raster(image_path, [1, 2, 3]), image_path, 
                                       Raster(label_path), label_path])

    def __getitem__(self, idx):
        image, image_path, label, _ = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image.getArray())
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=image.getArray())
            label = label.getArray()
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image.getArray(), label=label.getArray())
            return im, label

    def __len__(self):
        return len(self.file_list)
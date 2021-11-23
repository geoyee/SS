# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


import os.path as osp
import numpy as np
import math
from easydict import EasyDict as edict
from typing import List, Dict, Tuple, Union

try:
    from osgeo import gdal
except:
    import gdal


class Raster:
    def __init__(self, 
                 tif_path: str,
                 show_band: Union[List[int], Tuple[int]]=[1, 1, 1],
                 grid_size: Union[List[int], Tuple[int]]=[512, 512],
                 overlap: Union[List[int], Tuple[int]]=[24, 24]) -> None:
        """ 用于处理遥感栅格数据的类.
        参数:
            tif_path (str): GTiff数据的路径.
            show_band (Union[List[int], Tuple[int]], optional): 用于RGB合成显示的波段. 默认为 [1, 1, 1].
            grid_size (Union[List[int], Tuple[int]], optional): 切片大小. 默认为 [512, 512].
        """
        super(Raster, self).__init__()
        if osp.exists(tif_path):
            self.src_data = gdal.Open(tif_path)
            self.geoinfo = self.__getRasterInfo()
            self.show_band = list(show_band) if self.geoinfo.count != 1 else None
            self.grid_size = np.array(grid_size)
            self.overlap = np.array(overlap)
        else:
            raise("{0} not exists!".format(tif_path))

    def __getRasterInfo(self) -> Dict:
        geoinfo = edict()
        geoinfo.count = self.src_data.RasterCount
        geoinfo.xsize = self.src_data.RasterXSize
        geoinfo.ysize = self.src_data.RasterYSize
        geoinfo.gsr = self.src_data.GetSpatialRef()
        geoinfo.proj = self.src_data.GetProjection()
        geoinfo.geotf = self.src_data.GetGeoTransform()
        return geoinfo

    def setBand(self, bands: Union[List[int], Tuple[int]]) -> None:
        self.show_band = list(bands)

    def getArray(self) -> np.array:
        ima = self.src_data.ReadAsArray()
        if self.geoinfo.count == 3:
            ima = ima.transpose((1, 2, 0))
        return ima

    def getGrid(self, row: int, col: int) -> np.array:
        grid_idx = np.array([row, col])
        ul = grid_idx * (self.grid_size - self.overlap)
        lr = ul + self.grid_size
        ima = self.src_data.ReadAsArray(ul[1], ul[0], (lr[1] - ul[1]), (lr[0] - ul[0]))
        if self.geoinfo.count == 3:
            ima = ima.transpose((1, 2, 0))
        return ima

    def saveMask(self, img: np.array, save_path: str, 
                 geoinfo: Union[Dict, None]=None) -> None:
        if geoinfo is None:
            geoinfo = self.geoinfo
        driver = gdal.GetDriverByName('GTiff')
        datatype = gdal.GDT_Byte
        dataset = driver.Create(
            save_path, 
            geoinfo.xsize, 
            geoinfo.ysize, 
            geoinfo.count,
            datatype)
        dataset.SetProjection(geoinfo.proj)
        dataset.SetGeoTransform(geoinfo.geotf)
        C = img.shape[-1] if len(img.shape) == 3 else 1
        if C == 1:
            dataset.GetRasterBand(1).WriteArray(img)
        else:
            for i in range(C):
                dataset.GetRasterBand(i + 1).WriteArray(img[:, :, i])
        dataset.GetRasterBand(1).WriteArray(img)
        del dataset

    def saveMaskbyGrids(self, 
                        img_list: List[List[np.array]], 
                        save_path: Union[str, None]=None,
                        geoinfo: Union[Dict, None]=None) -> np.array:
        if geoinfo is None:
            geoinfo = self.geoinfo
        raw_size = (geoinfo.ysize, geoinfo.xsize)
        h, w = self.grid_size
        row = math.ceil(raw_size[0] / h)
        col = math.ceil(raw_size[1] / w)
        result_1 = np.zeros((h * row, w * col), dtype=np.uint8)
        result_2 = result_1.copy()
        for i in range(row):
            for j in range(col):
                ih, iw = img_list[i][j].shape[:2]
                im = np.zeros(self.grid_size)
                im[:ih, :iw] = img_list[i][j]
                start_h = (i * h) if i == 0 else (i * (h - self.overlap[0]))
                end_h = start_h + h
                start_w = (j * w) if j == 0 else (j * (w - self.overlap[1]))
                end_w = start_w + w
                if (i + j) % 2 == 0:
                    result_1[start_h: end_h, start_w: end_w] = im
                else:
                    result_2[start_h: end_h, start_w: end_w] = im
        result = np.where(result_2 != 0, result_2, result_1)
        result = result[:raw_size[0], :raw_size[1]]
        if save_path is not None:
            self.saveMask(result, save_path, geoinfo)
        return result
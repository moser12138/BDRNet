import numpy as np
from lib.data.base_dataset import BaseDataset
import lib.data.transform_cv2 as T


class CamVid(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CamVid, self).__init__(dataroot, annpath, trans_func, mode)

        # CamVid 数据集中有 11 个主要类别
        self.n_cats = 11  # 具体类别数，CamVid 通常用于 11 类的评估
        self.lb_ignore = 255  # 忽略的标签值，通常为 255
        self.lb_map = np.arange(256)  # 创建从 0 到 255 的标签映射

        # 均值和标准差通常需要根据数据集图像进行预处理。这里是一个典型的 RGB 均值和方差。
        # self.to_tensor = T.ToTensor(
        #     mean=(0.39068744, 0.40521396, 0.41455081),  # CamVid 数据集的 RGB 均值
        #     std=(0.29652027, 0.30514973, 0.30080369),  # CamVid 数据集的 RGB 标准差
        # )

        self.to_tensor = T.ToTensor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

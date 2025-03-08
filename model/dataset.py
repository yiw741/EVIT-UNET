import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import cv2
import random

class CustomDataset(Dataset):
    def __init__(self, root, split='train', num_classes=21, base_size=None, augment=True,
                 crop_size=321, scale=True, flip=True, rotate=True, blur=True):
        """
        初始化数据集类。
        :param root: 数据集根目录路径
        :param split: 数据集划分（如'train'或'val'）
        :param num_classes: 类别总数
        :param base_size: 基础输入图像大小
        :param augment: 是否进行数据增强
        :param crop_size: 裁剪后的图像大小
        :param scale: 是否进行缩放
        :param flip: 是否进行水平翻转
        :param rotate: 是否进行随机旋转
        :param blur: 是否进行高斯模糊
        """
        super(CustomDataset, self).__init__()
        self.root = root  # 数据集根路径
        self.num_classes = num_classes  # 类别总数
        self.base_size = base_size  # 基础图像大小
        self.crop_size = crop_size  # 裁剪图像大小
        self.augment = augment  # 是否进行数据增强
        self.split = split  # 数据集划分
        self.scale = scale  # 是否进行缩放
        self.flip = flip  # 是否进行翻转
        self.rotate = rotate  # 是否进行旋转
        self.blur = blur  # 是否进行模糊
        self.to_tensor = transforms.ToTensor()  # 转换为Tensor
        self._set_files()  # 设置文件路径

    def _set_files(self):
        """
        设置数据集中的图像和标签文件路径。
        """
        self.image_dir = os.path.join(self.root, 'images')  # 图像目录
        self.label_dir = os.path.join(self.root, 'imagesseg')  # 标签目录
        file_list = os.path.join(self.root, self.split + ".txt")  # 文件列表路径
        self.files = [line.rstrip() for line in open(file_list, "r")]  # 读取文件名列表

    def _load_data(self, index):
        """
        根据索引加载图像和标签。

        :param index: 图像索引
        :return: 图像和标签数组
        """
        image_id = self.files[index]  # 获取图像ID
        image_path = os.path.join(self.image_dir, image_id )  # 图像路径
        label_path = os.path.join(self.label_dir, image_id )  # 标签路径
        image = np.asarray(Image.open(image_path+'.jpg'), dtype=np.float32)  # 加载图像
        label = np.asarray(Image.open(label_path+'.png'), dtype=np.int32)  # 加载标签
            # 如果是灰度图像，将其转换为3通道
        if len(image.shape) == 2:  # 灰度图像
            image = np.stack([image] * 3, axis=-1)  # 复制到3个通道
        return image, label

    def __getitem__(self, index):
        """
        获取指定索引的图像和标签。

        :param index: 图像索引
        :return: 处理后的图像和标签
        """
        image, label = self._load_data(index)  # 加载数据
        if self.augment:  # 如果进行数据增强
            image, label = self._augmentation(image, label)  # 应用数据增强
        label = torch.from_numpy(np.array(label, dtype=np.float32)).long()  # 转换标签为Tensor
        image = Image.fromarray(np.uint8(image))  # 转换图像为PIL格式
        return self.to_tensor(image), label  # 返回图像和标签的Tensor

    def __len__(self):
        """
        返回数据集的长度。

        :return: 数据集样本数量
        """
        return len(self.files)

    def _augmentation(self, image, label):
        """
        应用数据增强方法。

        :param image: 输入图像
        :param label: 输入标签
        :return: 增强后的图像和标签
        """
        shape = image.shape
        if len(shape) == 2:  # 处理灰度图像
            h, w = shape
            channels = 1  # 假设是单通道图像
            image = image[:, :, np.newaxis]  # 将图像扩展为三维
        elif len(shape) == 3:  # 处理彩色图像
            h, w, channels = shape
        else:
            raise ValueError(f"Unexpected image shape: {shape}")

        # 进行缩放
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
                int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        # 随机旋转
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

        # 随机裁剪
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]  # 更新 h 和 w
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            image = image[start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]
            label = label[start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]

        # 随机水平翻转
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # 添加高斯模糊
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)

        # 如果是灰度图像，返回时需要去掉通道维度
        if channels == 1 and image.shape[-1] == 1:  # 确保最后一个维度为1
            image = image.squeeze(-1)  # 去掉最后一个维度

        return image, label  # 返回增强后的图像和标签
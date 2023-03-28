# dataset.py采用了pytorch的Dataset和DataLoader，用于加载数据集和批量读取数据
# 同时使用了natsort对文件名进行排序，以保证文件名的顺序和文件夹中的顺序一致，避免出现错误。
# 由于不确定数据集中的图片格式，所以在读取图片时，需要将图片转换为RGB格式，否则会报错。

import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted

# 将图片从所指定的地址复制到rgb_image中，然后返回rgb_image
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

# torch.utils.data.Dataset是一个表示数据集的抽象类
# 你自己的数据集应该继承Dataset，并且重写__len__和__getitem__两个方法
class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


transform = T.Compose([
    T.RandomHorizontalFlip(), # 随机水平翻转
    T.RandomVerticalFlip(), # 随机垂直翻转
    T.RandomCrop(c.cropsize), # 随机裁剪
    T.ToTensor() # 将图片转换为Tensor,归一化至[0,1]
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val), # 中心裁剪
    T.ToTensor(), # 将图片转换为Tensor,归一化至[0,1]
])


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"), # 加载数据集
    batch_size=c.batch_size, # 每个batch加载多少个样本
    shuffle=True, # 是否打乱数据
    pin_memory=True, # 将数据保存在pin memory区，然后在GPU中复制
    num_workers=8, # 读取数据的进程数
    drop_last=True # 如果数据集大小不能被batch size整除，是否丢弃最后一个不完整的batch
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)
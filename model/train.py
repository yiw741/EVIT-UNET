import torch
from torch import optim
from tqdm import tqdm # 进度条显示
from torch.utils.data import DataLoader
from dataset import CustomDataset
from My_test.My_evit_unet.model import Eff_Unet
from  helper_more import eval_metrics
import numpy as np
import os

# 判断能否使用gpu加速运算
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_root = r'D:\data_demo\demo_voc_test'  # 填写数据集的路径
# 在脚本开始添加目录创建代码
save_dir = r'./weights'
os.makedirs(save_dir, exist_ok=True)  # 自动创建目录，如果目录已存在则不会报错

EPOCH = 10
num_classes = 21
batch_size = 15
pre_val = 2
crop_size = 224  # 调整为常用尺寸

# 实例化 daloader，填写自己的数据集地址
train_datasets = CustomDataset(root=data_root,split='train',num_classes=num_classes,base_size=300,crop_size=crop_size,scale=True,flip=True,rotate=True)
val_datasets = CustomDataset(root=data_root,split='val',num_classes=num_classes,base_size=300,crop_size=crop_size,scale=True,flip=True,rotate=True)
train_dataloader = DataLoader(train_datasets,batch_size=batch_size,num_workers=1,shuffle=True,drop_last=True)
val_dataloader = DataLoader(val_datasets,batch_size=batch_size,num_workers=1,shuffle=True,drop_last=True)




# 实例化模型
model = Eff_Unet(
    layers=[5, 5, 15, 10],
    embed_dims=[40, 80, 192, 384],
    downsamples=[True, True, True, True],
    vit_num=6,
    drop_path_rate=0.1,
    num_classes=num_classes,
    fork_feat=True
)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # 初始学习率
    weight_decay=1e-4   # L2正则化
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,   # 每10个epoch学习率衰减
    gamma=0.5       # 学习率乘以0.5
)
# 损失函数
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(epoch):
    total_loss = 0
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    model.to(device)
    model.train()

    tbar = tqdm(train_dataloader, ncols=130)
    for index, (image, label) in enumerate(tbar):
        image = image.to(device)
        label = label.to(device)

        # 前向传播
        output = model(image)

        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        lr = get_lr(optimizer)

        total_loss += loss.item()

        seg_metrics = eval_metrics(output, label, num_classes)
        correct, num_labeled, inter, union = seg_metrics

        total_correct += correct
        total_label += num_labeled
        total_inter += inter
        total_union += union

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        tbar.set_description(
            'TRAIN {}/{} | Loss: {:.3f}| Acc {:.2f} mIoU {:.2f} | lr {:.8f}|'.format(
                epoch, EPOCH,
                np.round(total_loss/(index+1),3),
                np.round(pixAcc,3),
                np.round(mIoU,3),
                lr
            )
        )

    return total_loss / (index + 1)

def val_epoch(epoch):
    total_loss = 0
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    model.to(device)
    model.eval()

    print(f'正在使用 {device} 进行验证! ')
    tbar = tqdm(val_dataloader, ncols=130)

    with torch.no_grad():
        for index, (image, label) in enumerate(tbar):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = loss_fn(output, label)

            total_loss += loss.item()

            seg_metrics = eval_metrics(output, label, num_classes)
            correct, num_labeled, inter, union = seg_metrics

            total_correct += correct
            total_label += num_labeled
            total_inter += inter
            total_union += union

            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            tbar.set_description(
                'EVAL ({})|Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}|'.format(
                    epoch,
                    total_loss/(index+1),
                    pixAcc,
                    mIoU
                )
            )

        print('Finish validation!')
        print(f'total loss:{np.round(total_loss/(index+1),3)} || PA:{np.round(pixAcc,3)} || mIoU:{np.round(mIoU,3)}')
        print(f'every class Iou {dict(zip(range(num_classes), np.round(IoU,3)))}')

        print('正在保存权重！！！！')
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = os.path.join(save_dir, f'checkpoint--epoch{epoch}.pth')
        torch.save(state, filename)
        print(f'成功保存第{epoch}epoch权重文件')

    return total_loss / (index + 1), {
        'pixAcc': pixAcc,
        'mIoU': mIoU
    }

def train(EPOCH):
    print(f'正在使用 {device} 进行训练! ')
    for i in range(EPOCH):
        train_loss = train_epoch(i)
        if i % pre_val == 0:
            val_loss, val_metrics = val_epoch(i)

if __name__ == '__main__':
    train(EPOCH)

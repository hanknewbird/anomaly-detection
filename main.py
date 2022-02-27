from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Grayscale
import argparse
import torch.nn.functional as F
from torch.optim import Adam
import torch
from trainer import train
from model_plus import NewNet
from datetime import datetime

# --train_dir Class9_dataset/train/normal
# --val_dir Class9_dataset/test/normal
# tensorboard -–logdir /path/to/logs


def create_data_tensor(data_dir, batch_size=8):
    # 定义数据预处理操作：将图像转化为灰度图像，将其转化为tensor格式
    transform = Compose([Grayscale(), ToTensor()])
    # 使用以上定义的预处理方式读取数据集
    dataset = ImageFolder(data_dir, transform=transform)
    # 定义一个数据迭代加载器
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, help="训练集路径")
    parser.add_argument('--val_dir', required=True, help="验证集路径")
    parser.add_argument("--log_interval", type=int, default=1, help="每1次训练记录一次")
    parser.add_argument('--epochs', type=int, default=40, help="epochs")
    parser.add_argument('--train_batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--val_batch_size', type=int, default=4, help="val batch size")
    parser.add_argument("--log_dir", type=str, default=f'tensorboard_logs_{datetime.now().strftime("%d%m%Y_%H-%M")}', help="日志保存路径")
    parser.add_argument('--load_weight_path', type=str, help="需要加载的权重路径")
    parser.add_argument('--save_graph', action='store_true', help="保存图")
    args = parser.parse_args()

    # 使用Adam优化器
    optimizer = Adam
    # 如果设备没有GPU就用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 使用MSE作为损失函数
    loss = F.mse_loss
    # 训练集
    train_loader = create_data_tensor(args.train_dir, args.train_batch_size)
    # 测试集
    val_loader = create_data_tensor(args.val_dir, args.val_batch_size)
    # 设置模型为自定义网络结构
    model = NewNet()
    # 训练开始
    train(model, optimizer, loss, train_loader,
          val_loader, args.log_dir, device, args.epochs,
          args.log_interval,
          args.load_weight_path, args.save_graph)

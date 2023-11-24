# 项目来源 https://github.com/pytorch/examples/tree/main/mnist

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3x3，步长为1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1
        self.dropout1 = nn.Dropout(0.25)  # 定义一个Dropout层，丢弃输入中25%的元素
        self.dropout2 = nn.Dropout(0.5)  # 定义一个Dropout层，丢弃输入中50%的元素
        self.fc1 = nn.Linear(9216, 128)  # 定义第一个全连接层，输入维度为9216，输出维度为128
        self.fc2 = nn.Linear(128, 10)  # 定义第二个全连接层，输入维度为128，输出维度为10

    def forward(self, x):
        '''
        前向传播函数
        参数:
            x: 输入数据
        返回值:
            output: 经过网络前向传播后的输出
        '''
        x = self.conv1(x)  # 使用第一个卷积层
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.conv2(x)  # 使用第二个卷积层
        x = F.relu(x)  # 使用ReLU激活函数
        x = F.max_pool2d(x, 2)  # 进行最大池化操作
        x = self.dropout1(x)  # 进行随机失活操作
        x = torch.flatten(x, 1)  # 将张量展平
        x = self.fc1(x)  # 使用第一个全连接层
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.dropout2(x)  # 进行随机失活操作
        x = self.fc2(x)  # 使用第二个全连接层
        output = F.log_softmax(x, dim=1)  # 使用log_softmax激活函数
        return output  # 返回输出结果


def train(model, device, train_loader, optimizer, epoch):
    """
    训练函数
    
    参数:
        model: 训练模型
        device: 设备（GPU或CPU）
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前的训练轮数
    
    返回值:
        train_loss: 平均训练损失
        train_accuracy: 训练准确率
    """
    model.train()  # 设置模型为训练模式
    train_loss = 0  # 初始化训练损失
    correct = 0  # 初始化正确预测数量
    total = len(train_loader.dataset)  # 训练数据总数量
    
    # 创建训练进度条，并设置显示轮数、数据加载器、描述和 leave 参数为False
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (data, target) in progress_bar:  # 遍历训练数据
        data, target = data.to(device), target.to(device)  # 将数据和目标转移到指定设备上
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 模型前向传播
        loss = F.nll_loss(output, target)  # 计算损失
        train_loss += loss.item()  # 累加训练损失
        pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
        correct += pred.eq(target.view_as(pred)).sum().item()  # 累加正确预测数量
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        
        # 更新进度条的描述
        progress_bar.set_description(f'Epoch {epoch} [Loss: {loss.item():.6f}]')
    
    train_loss /= total  # 计算平均训练损失
    train_accuracy = 100. * correct / total  # 计算训练准确率
    print(f'Train Epoch: {epoch} Average loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')  # 打印训练结果
    return train_loss, train_accuracy  # 返回训练损失和准确率

def test(model, device, test_loader):
    """
    用于测试模型的函数

    参数:
        model: 要测试的模型
        device: 设备（CPU或GPU）
        test_loader: 用于加载测试数据的DataLoader

    返回值:
        test_loss: 测试集上的平均损失
        test_accuracy: 测试集上的准确率
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加批量损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n')
    return test_loss, test_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='训练时的批量大小 (默认: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='测试时的批量大小 (默认: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='训练的轮数 (默认: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='学习率 (默认: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='学习率步进参数 gamma (默认: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA训练')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='禁用macOS GPU训练')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='快速检查单次通过')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='随机种子 (默认: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='训练状态记录间隔 (默认: 10)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='保存当前模型')
    args = parser.parse_args()


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()


    # 设置随机种子
    torch.manual_seed(args.seed)


    # 设置设备
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    # 设置训练和测试的batch_size
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    # 设置数据变换
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    # 获取当前目录的绝对路径
    current_dir = os.getcwd() # 获取当前目录的绝对路径
    print(current_dir)
    data_dir = os.path.join(current_dir, 'data') # 将当前目录和'data'拼接
    print(data_dir)


    # 加载训练集和测试集
    dataset1 = datasets.MNIST(data_dir, train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST(data_dir, train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    # 初始化模型
    model = Net().to(device)


    # 设置优化器
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    # 设置学习率调整策略
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []


    # 开始训练
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        scheduler.step()


    # 绘制损失和准确率图表
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='训练损失')
    plt.plot(range(1, args.epochs + 1), test_losses, label='测试损失')
    plt.title('训练和测试损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), train_accuracies, label='训练准确率')
    plt.plot(range(1, args.epochs + 1), test_accuracies, label='测试准确率')
    plt.title('训练和测试准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # 保存模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
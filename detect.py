import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os

# 假设您的网络模型类定义如下
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

def load_model(model_path):
    """
    加载预训练模型

    参数:
        model_path (str): 模型文件的路径

    返回:
        model (Net): 加载的模型对象
    """
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image_path, model):
    """对单个图像进行预测"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(),  # 确保图像是灰度的
        transforms.Resize((28, 28)),  # 调整图像大小以匹配 MNIST 样本
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的标准化
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 预测
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)
        return prediction.item()

def main():
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(description='MNIST检测')
    
    # 添加一个参数，指定图像文件的路径，默认为'test.jpg'
    parser.add_argument('--image', type=str, default='test.jpg' , help='图像文件的路径')
    
    # 添加一个参数，指定模型文件的路径，默认为'mnist_cnn.pt'
    parser.add_argument('--model', type=str, default='mnist_cnn.pt', help='模型文件的路径')
    
    # 解析命令行参数
    args = parser.parse_args()

    # 检查图像文件是否存在，如果不存在则抛出文件未找到的异常
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"{args.image} 未找到")
    
    # 加载模型
    model = load_model(args.model)
    
    # 对图像进行预测
    prediction = predict_image(args.image, model)
    
    # 打印预测结果
    print(f'预测图像 {args.image} 的标签为：{prediction}')

if __name__ == "__main__":
    main()

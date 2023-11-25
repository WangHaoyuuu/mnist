import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

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
    """加载预训练模型"""
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image):
    """对单个图像进行预测"""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)
        return prediction.item()

# 加载模型
model_path = 'mnist_cnn.pt'  # 模型文件路径
model = load_model(model_path)




# 创建 Gradio 接口
iface = gr.Interface(
    fn=predict_image,
    inputs="image",  # 直接使用字符串来定义输入类型
    outputs="text",
    title="MNIST Digit Recognition",
    description="Upload an image of a handwritten digit and see the model prediction.",
    # CSS样式，用于调整图像显示大小width: 200px; height: 200px;
    # css = "width: 200px; height: 200px;", 
    # Gradio的图像输入组件在提交之前可能不会保持固定的大小。这是因为Gradio的图像输入组件会根据上传的图像的尺寸自动调整显示大小。
    examples=[['test.jpg'], 
              ['test1.jpg'], 
              ['test2.jpg']]  # 指定示例图片
)

# 运行接口
if __name__ == "__main__":
    iface.launch()

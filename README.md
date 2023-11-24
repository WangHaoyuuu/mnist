# Image Classification Using ConvNets
# 使用卷积神经网络进行图像分类

## 目录介绍
- data文件夹
  - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
  - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
  - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
  - Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)
  - train 文件夹是经过 convert.py 转换后的图片
  - test 文件夹是经过 convert.py 转换后的图片
  - train.txt 是经过 convert.py 转换后的图片对应的标签
  - test.txt 是经过 convert.py 转换后的图片对应的标签
- convert.py : 用于将以字节的形式进行存储的数据转换为可见的形式
- main.py : 项目主体文件
- verify_torch_GPU.py : 用于检测 GPU 是否可用
- 环境配置.py 所有相关 conda 和 pytorch 以及 pip 安装
- requirements.txt 所有相关 pip 安装
- detect.py : 用于检测图片的标签

## 环境安装
- python环境
  - conda create -n mnist python=3.8
- pytorch相关(参考环境配置中)
  - conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 cudnn=8.2.1 -c pytorch
- pip 安装(已导出到 requirements.txt ) 
  - pip install tqdm
  - pip install scikit-image
  - pip install matplotlib
  - 也可以执行 pip install -r requirements.txt 
- 关于 conda 和 pip 安装
  - 尽量先使用 conda 安装相关包，如果安装不成功，再使用 pip 安装。因为conda会自动解决包之间的依赖关系，而 pip 不会。

## 数据集简介
MNIST数据集是一个广泛使用的手写数字识别数据集，它包含了0到9共十个数字的手写样本。这个数据集由美国国家标准与技术研究院（National Institute of Standards and Technology, NIST）创建，并经过修改以便于在机器学习领域中的使用。它分为两个部分：训练集和测试集。
训练集：包含60,000个样本，用于训练模型。
测试集：包含10,000个样本，用于评估模型的性能。
每个样本都是一个28x28像素的灰度图像，其中每个像素的值介于0到255之间。这些图像被标准化并居中，以确保一致性。每个图像都有一个与之对应的标签，表示图像中的数字（0到9）。  

官方网址:<http://yann.lecun.com/exdb/mnist/>  
实测很多浏览器会跳转成<https://yann.lecun.com/exdb/mnist/>  
将https改成http就可访问

## 卷积神经网络用于mnist数据集简介
卷积神经网络（Convolutional Neural Networks, CNNs）在应用于MNIST数据集时，通常涉及以下几个关键步骤和组件：

1. 卷积层（Convolutional Layers）：这些层使用一组可学习的过滤器（或称为卷积核）来提取图像中的特征。每个过滤器负责检测图像的某些特征，如边缘、角点或更复杂的图案。

2. 激活函数：通常在卷积层之后应用非线性激活函数，如ReLU（Rectified Linear Unit）。这有助于网络学习复杂的模式。

3. 池化层（Pooling Layers）：这些层用于降低图像的空间尺寸（宽度和高度），以减少计算量和参数的数量。最常用的池化操作是最大池化（Max Pooling），它提取特定窗口内的最大值。

4. 全连接层（Fully Connected Layers）：在多个卷积和池化层之后，网络通常包含一个或多个全连接层。这些层的目的是将之前学到的特征表示映射到最终的输出类别（在MNIST的情况下为10个数字类别）。

5. 输出层：对于MNIST，输出层通常是一个具有10个神经元的全连接层，每个神经元对应一个数字类别（0-9）。输出层通常使用softmax激活函数，它将网络的输出转换为概率分布。

6. 损失函数：在训练CNN时，通常使用交叉熵损失函数（cross-entropy loss），它衡量模型输出的概率分布与真实标签的概率分布之间的差异。

7. 优化算法：用于调整网络参数以最小化损失函数。常用的优化算法包括随机梯度下降（SGD）和其变体，如Adam。

当应用于MNIST数据集时，CNN通常能够达到非常高的准确率，这使得它成为处理此类图像识别任务的流行选择。


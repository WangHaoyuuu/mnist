# 此处的环境采用的是李沐老师的  d2l
# 这样正好一本书《动手学习深度学习》也可以用

# 从conda开始
"""
    conda安装
"""
# 建议使用miniconda是因为它是一个轻量级的conda发行版，只包含conda、Python和一些必要的包，
# 而不像Anaconda那样包含大量的预装软件包，这使得miniconda更加灵活和易于管理。

# 国内可使用清华镜像下载
# https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

# 我选择的版本
# Miniconda3-py38_4.8.3-Windows-x86_64
# 没有特别的什么理由

# 安装时注意一个点
# 会出现
# Add miniconda3 to my PATH environment variable 
# 如果是长时间使用conda来管理python环境建议勾选(我一直都直接勾选了，没有发现什么问题)
# 这样在命令行中就可以直接使用conda命令了，而不需要每次都输入miniconda3的安装路径。

# 验证conda
# conda -V(大写)
# 出现conda版本即表示安装成功

# conda 23.3.1

"""
    conda换源
"""
# 更新至国内下载源(windows系统)
# 原因：This can significantly speed up the download process and reduce the chance of errors occurring during the download.
# 生成.condarc文件：这个文件是用来配置conda的下载源的
# conda config --set show_channel_urls yes

# 查看.condarc路径
# 执行 conda info
# 输出中的 user config file : C:\Users\WHY\.condarc 即是路径
# 在命令行输入 C:\Users\WHY\.condarc (按照查看的来) 打开该文件并内容改为

# 引号是注释(只复制内容)
"""

channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

"""
# 引号是注释(只复制内容)

# 保存后推出并执行
# conda clean -i
# 换源即完成

"""
    创建并激活环境和下载必要包
"""

# conda create --name d2l python=3.9 -y
# -y标志用于自动确认软件包的安装，而不需要用户输入。
# 此命令将创建一个名为d2l的conda环境，并安装Python 3.9。

# 激活conda环境
# conda activate d2l 

############### 使用CPU ###############
# 执行代码
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
# 验证是否安装成功CPU版本测试代码在  3.py
# 出现以下提示
# 1.12.0
# False
# None

############### 使用GPU ###############

# 这里有一个巨坑!!!

# 建议看这个blog就能理解了
# https://windses.blog.csdn.net/article/details/125910538

# 以下是2023年4月7日测试可用
# pytorch-1.12.0-py3.9_cuda11.3_cudnn8_0.tar.bz2 是可用的组合包

# 执行代码
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 cudnn=8.2.1 -c pytorch

# 安装了PyTorch版本1.12.0、torchvision版本0.13.0、torchaudio版本0.12.0和cudatoolkit版本11.3。
# PyTorch是基于Torch库的开源机器学习库，
# torchvision是一个包，为PyTorch中的计算机视觉任务提供了流行的数据集、模型架构和图像转换,
# torchaudio是一个包，为PyTorch提供音频处理功能。它包括支持加载和预处理音频数据以及构建和训练音频模型的功能。
# cudatoolkit是用于加速使用NVIDIA GPU的应用程序的软件开发工具包（SDK）。
# cuDNN是CUDA深度神经网络库的缩写，是NVIDIA针对深度学习任务的加速库。
# 它是基于CUDA的，提供了高效的卷积、池化、归一化等操作的实现。
# 因此，cuDNN可以与CUDA一起使用，加速深度学习模型的训练和推理过程。

# 验证是否安装成功GPU版本测试代码
# 验证torch是否安装完成
# import torch
# print(torch.__version__)
# # 验证cuda
# print(torch.cuda.is_available())
# # 查看CUDA版本
# print(torch.version.cuda)

# 出现以下提示
# 1.12.0
# True
# 11.3
# 成功11111 !

# 至此,所用的环境配置已经完成

"""
    pip 命令 
"""

# pip 换源
# python -m pip install --upgrade pip
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 使python环境能用于jupyter notebook
# pip install ipykernel

# 李沐老师的包
# pip install d2l


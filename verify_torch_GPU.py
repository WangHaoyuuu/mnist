import torch
print(torch.__version__)
# 验证cuda
print(torch.cuda.is_available())
# 查看CUDA版本
print(torch.version.cuda)
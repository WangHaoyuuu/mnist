# 参考 https://www.cnblogs.com/denny402/p/7520063.html
# 优化了路径获取， 函数表述和进度条显示

from pathlib import Path
from skimage import io
import torchvision.datasets.mnist as mnist
from tqdm import tqdm

# 获取当前目录的绝对路径
current_dir = Path.cwd()
data_dir = current_dir / 'data' / 'MNIST' /'raw'
print(data_dir)

train_set = (
    mnist.read_image_file(data_dir / 'train-images-idx3-ubyte'),
    mnist.read_label_file(data_dir / 'train-labels-idx1-ubyte')
)

test_set = (
    mnist.read_image_file(data_dir / 't10k-images-idx3-ubyte'),
    mnist.read_label_file(data_dir / 't10k-labels-idx1-ubyte')
)

def convert_to_img(dataset, dataset_type='train'):
    """
    将MNIST图像转换为JPEG并保存标签到txt文件中。
    :param dataset: 图像和标签的元组。
    :param dataset_type: 'train'或'test'来指定数据集类型。
    """
    data_path = data_dir / dataset_type
    data_path.mkdir(parents=True, exist_ok=True)

    label_file = data_dir / f'{dataset_type}.txt'
    with label_file.open('w') as f:
        # 包装循环以显示进度条
        for i, (img, label) in enumerate(tqdm(zip(*dataset), total=len(dataset[0]), desc=f'处理 {dataset_type} 图像')):
            img_path = data_path / f'{i}.jpg'
            io.imsave(str(img_path), img.numpy())
            f.write(f'{img_path} {label}\n')

print(f"Training set size: {train_set[0].size()}")  # 打印训练集的大小
print(f"Test set size: {test_set[0].size()}")  # 打印测试集的大小

convert_to_img(train_set, 'train')  # 将训练集转换为图像
convert_to_img(test_set, 'test')  # 将测试集转换为图像

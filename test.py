from ss.datasets import RSDataset
import paddleseg.transforms as T


# 构建训练集
train_transforms = [
    T.RandomHorizontalFlip(),
    T.RandomRotation(),
    T.Resize(target_size=(256, 256))
]
train_dataset = RSDataset(
    transforms=train_transforms,
    dataset_root='DataSet',
    num_classes=2,
    mode='train',
    file_path='DataSet/train_list.txt',
    separator=' ',
    big_map=False
)

for img, lab in train_dataset:
    print(img.shape, lab.shape)
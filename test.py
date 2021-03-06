from ss.datasets import RSDataset
import paddleseg.transforms as T

# DEBUG
import cv2


def img2show(img):
    img = img.transpose([1, 2, 0]).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


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
    # file_path='DataSet/train_list.txt',
    file_path='DataSet/train_list_2.txt',
    separator=' ',
    big_map=True  # False
)

lens = len(train_dataset)
print(f"lens={lens}")
for idx, (img, lab) in enumerate(train_dataset):
    print(idx, img.shape, lab.shape)
    cv2.imshow("img", img2show(img))
    cv2.imshow("lab", lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import os
import json
import argparse
import sys

import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from prettytable import PrettyTable

from utils import read_split_data, read_test_data
from my_dataset import MyDataSet
from models.ConvBiFuseNet import ConvBiFusion as creatmodel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['axes.unicode_minus'] = False

class Mixup(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img

        beta = np.random.beta(self.alpha, self.alpha)
        beta = max(beta, 1.0 - beta)

        img_size = img.size
        img2 = F.to_pil_image(np.array(img))

        img = F.to_pil_image(np.array(img))
        img = Image.blend(img, img2, beta)

        return img

class Cutout(object):
    def __init__(self, num_holes=8, max_h_size=8, max_w_size=8, fill_value=0):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img

        img_size = img.size
        h = img_size[1]
        w = img_size[0]

        for _ in range(self.num_holes):
            h_loc = np.random.randint(0, h)
            w_loc = np.random.randint(0, w)
            h_size = np.random.randint(1, self.max_h_size)
            w_size = np.random.randint(1, self.max_w_size)

            h1 = max(0, h_loc - h_size // 2)
            h2 = min(h, h_loc + h_size // 2)
            w1 = max(0, w_loc - w_size // 2)
            w2 = min(w, w_loc + w_size // 2)

            img = np.array(img)
            img[h1:h2, w1:w2] = self.fill_value
            img = Image.fromarray(img)

        return img

class Mosaic(object):
    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img

        img_size = img.size
        img2 = F.to_pil_image(np.array(img))

        h, w = img_size[1], img_size[0]
        yc, xc = [int(np.random.uniform(0.4 * h, 0.6 * h)) for _ in range(2)]  # Mosaic center coordinates
        h_factor, w_factor = [np.random.uniform(0.3, 0.7) for _ in range(2)]  # Mosaic scaling factors

        # Create mosaic image
        img.paste(img2.resize((int(h * h_factor), int(w * w_factor))), (xc, yc))

        return img

class ConfusionMatrix(object):
    """
    Note: If the displayed images are incomplete, it may be an issue with the version of Matplotlib.
    This example works correctly with Matplotlib 3.2.1 (on Windows and Ubuntu).
    Additionally, the PrettyTable library needs to be installed.
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity, f1 score
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1 Score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1_Score = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1_Score])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('MRI2_matrix')
        plt.show()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # _, _, val_images_path, val_images_label = read_split_data(args.data_path)
    test_images_path, test_images_label = read_test_data(args.data_path)


    img_size = 224
    data_transform = {
        # "val": transforms.Compose([transforms.Resize((img_size, img_size)),
        #                            transforms.CenterCrop(img_size),
        #                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        #                            transforms.RandomHorizontalFlip(),

        #                            transforms.ToTensor(),
        #                            transforms.Normalize([0.56490765, 0.54041824, 0.78989806],
        #                                                 [0.14042906, 0.12497939, 0.09305927])])}
        "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                   transforms.RandomHorizontalFlip(),

                                   transforms.ToTensor(),
                                   transforms.Normalize([0.33711524, 0.33711524, 0.33711524], [0.29203557, 0.29203557, 0.29203557])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = creatmodel(num_classes=args.num_classes)
    # load pretrain weights
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/home/s11/LSG/DATA/MRI2")

    # 训练权重路径
    parser.add_argument('--weights', type=str, default='/home/s11/LSG/C-Tnet/weights/MRI2(0).pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
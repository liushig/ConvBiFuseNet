import os
import argparse
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
from torch import optim as optim
from my_dataset import MyDataSet
from models.ConvBiFuseNet import ConvBiFuseNetatto as creatmodel
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False


def main(args):
    device = torch.device(args.device)
    # if torch.cuda.is_available() else "cpu"
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.Resize((img_size, img_size)),
                                     transforms.CenterCrop(img_size),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.33711524, 0.33711524, 0.33711524],
                                                          [0.29203557, 0.29203557, 0.29203557])]),

        "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.33711524, 0.33711524, 0.33711524],
                                                        [0.29203557, 0.29203557, 0.29203557])])}

    # mean: [0.56490765, 0.54041824, 0.78989806]
    # std: [0.14042906, 0.12497939, 0.09305927]

    # mean: [0.33878461, 0.33878461, 0.33878461]
    # std: [0.29483773, 0.29483773, 0.29483773]

    mean: [0.33711524, 0.33711524, 0.33711524]
    std: [0.29203557, 0.29203557, 0.29203557]

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = creatmodel(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # Remove weights related to classification head
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Freeze all weights except for the head
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # Get parameters for optimizer
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    def matplot_loss(train_loss, val_loss):
        plt.plot(val_loss, label='val_loss')
        plt.plot(train_loss, label='train_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.title("Comparison graph of loss values between the training set and the validation set")
        plt.legend(loc='best')
        plt.savefig('Loss-curveMRI2(MRI2PLUS)P.png')
        plt.show()

    def matplot_acc(train_acc, val_acc):
        plt.plot(val_acc, label='val_acc')
        plt.plot(train_acc, label='train_acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(loc='best')
        # plt.title("")
        plt.savefig('Acc-curveMRI2(MRI2PLUS)P.png')
        plt.show()

    best_acc = 0.
    for epoch in range(args.epochs):
        # for inputs, labels in train_loader:
        #     print("Example Target Labels:", labels)

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        acc_train.append(train_acc)
        loss_train.append(train_loss)
        acc_val.append(val_acc)
        loss_val.append(val_loss)

        print("[epoch {}] train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}".format(
            epoch + 1, train_loss, train_acc, val_loss, val_acc))
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "weights/trian-convbifuse-MRI2PLUS(1).pth")
            best_acc = val_acc

    matplot_loss(loss_train, loss_val)
    matplot_acc(acc_train, acc_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=1e-4)

    # Root directory of the dataset
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/home/s11/LSG/DATA/MRI2PLUS")

    parser.add_argument('--weights', type=str, default='weights/trian-convbifuse-MRI2PLUS(0).pth',
                        help='initial weights path')
    # Whether to freeze all weights except the head
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)

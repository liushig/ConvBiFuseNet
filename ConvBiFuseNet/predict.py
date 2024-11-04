import os
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.ConvBiFuseNet import ConvBiFuseNetatto as create_model
from pandas.core.frame import DataFrame

def predict_folder(folder_path, model, class_indict, device):
    # Get all image file names in the folder
    image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]

    # Initialize a list to store results
    predictions = []
    image_names = []

    # Data preprocessing
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.56490765, 0.54041824, 0.78989806], [0.14042906, 0.12497939, 0.09305927]),  # Chest X-ray (Covid-19 & Pneumonia)
    ])

    # Predict for each image
    model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)

            # Read the image
            img = Image.open(img_path).convert('RGB')
            image_names.append(os.path.basename(img_path))

            # Data preprocessing
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            # Inference
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

            # Get the predicted class
            predictions.append(class_indict[str(predict_cla)][:-1])

    return image_names, predictions

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 3

    # Create the model
    model = create_model(num_classes=num_classes).to(device)

    # Load the pretrained weights
    model_weight_path = "/home/s11/LSG/C-Tnet/weights/data(1).pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # Read the class label mapping
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Specify the path of the folder to predict
    folder_path = "/home/s11/LSG/DATA/MRI1_data/test/meningioma"

    # Make predictions
    image_names, predictions = predict_folder(folder_path, model, class_indict, device)

    # Save the results in a CSV file
    c = {"Image Name": image_names, "Predicted Class": predictions}
    data = DataFrame(c)
    outputpath = 'results.csv'
    data.to_csv(outputpath, sep=',', index=False)

if __name__ == '__main__':
    main()

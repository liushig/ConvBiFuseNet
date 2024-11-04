ConvBiFuseNet: A Parallel Fusion Model with Routing Attention for MRI Brain Tumor Classification


This repository contains the implementation of ConvBiFuseNet, a novel network architecture that combines convolutional layers with bi-level routing attention to improve classification accuracy on medical imaging data. The model integrates convolutional features and attention mechanisms, allowing it to capture spatial and contextual information in a balanced way.

[模型图(竖版).pdf](https://github.com/user-attachments/files/17617804/default.pdf)

Model Architecture
The ConvBiFuseNet architecture utilizes convolutional blocks (ConvBraNet) and Bi-level Routing Attention (BRA) blocks in parallel branches. Key components include:

DWCov (Depthwise Convolution): Applied at the input layer for efficient feature extraction.
ConvBraNet Block: Processes convolutional features, performing down-sampling and dimensionality adjustments at each stage.
BiFormer Block: A Transformer-based module with patch embedding, which captures contextual information over larger spatial regions.
Fusion Mechanism: Linear fusion layers are used to combine features from both branches at each level, followed by a final linear layer for classification.
Diagram

Files

ConvBiFuseNet.py: Defines the main ConvBiFuseNet model, combining ConvBraNet and BiFormer branches, and implementing fusion layers to integrate their outputs.
ConvBraNet.py: Contains the definition of the ConvBraNet block, which is a convolutional network used for feature extraction in the model.
Dataset
The model is designed for medical imaging datasets, particularly for MRI-based classification tasks. Make sure your dataset follows the expected structure, with separate folders for each class.

Model Components

ConvBraNet Block: Uses convolutional layers with Grouped and Layer Normalization to capture spatial information.
BiFormer Block: A Transformer-based module with multi-head self-attention and patch embedding, capturing contextual relationships.
Bi-level Routing Attention (BRA): Allows the network to focus on relevant regions at multiple levels, enhancing feature representation.
Fusion Layers: Linearly combines outputs from the convolutional and attention-based branches.

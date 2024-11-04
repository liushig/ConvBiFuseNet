import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from models.ConvBraNet import ConvBraNet, LayerNorm
from models.biformer_stl_nchw import nchwBiFormerSTL
from timm.models.vision_transformer import _cfg


class ConvBiFusion(nn.Module):
    def __init__(self, ConvBraNet_model, biformer_model, num_classes, dims=[40, 80, 160, 320]):
        super(ConvBiFusion, self).__init__()
        self.ConvBraNet_downsample_layers = ConvBraNet.downsample_layers
        self.convnext_stages = ConvBraNet.stages
        self.biformer_downsample_layers = biformer_model.downsample_layers
        self.biformer_stages = biformer_model.stages
        self.num_classes = num_classes
        self.fusion_weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(4)])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
    def forward(self, x):
        convnext_features = []
        biformer_features = []
        fusion_weights = []

        for i in range(4):
            initial_x = x  # Saving the initial input value
            # print(f'ConvNeXt Stage input {i + 1}  size:', x.size())
            x = self.convnext_downsample_layers[i](x)
            x = self.convnext_stages[i](x)
            # print(f'convnext_stages output {i + 1} size:', x.size())
            convnext_features.append(x)

            x_bi = initial_x
            x_bi = self.biformer_downsample_layers[i](x_bi)
            x_bi = self.biformer_stages[i](x_bi)
            biformer_features.append(x_bi)
            convnext_output = convnext_features[i]
            biformer_output = biformer_features[i]
            fused_feature = self.fusion_weights[i] * convnext_output + (1 - self.fusion_weights[i]) * biformer_output
            x = fused_feature + convnext_output

            fusion_weights.append(self.fusion_weights[i])


        fused_output = self.norm(x.mean([-2, -1]))
        classification_scores = self.head(fused_output)
        return classification_scores
        # return classification_scores, fusion_weights



def ConvBiFuseNetatto(pretrained=False, pretrained_cfg=None,
                               pretrained_cfg_overlay=None, **kwargs):
    ConvBraNet_model = ConvBraNet()
    biformer_model = nchwBiFormerSTL()

    model = ConvBiFusion(
        convnext_model=ConvBraNet_model,
        biformer_model=biformer_model,
        **kwargs)
    model.default_cfg = _cfg()

    return model
# print(ConvNeXtBiFormerFusionatto)

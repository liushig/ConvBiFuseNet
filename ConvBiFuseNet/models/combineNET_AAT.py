import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ConvBraNet import ConvBraNet, LayerNorm
from models.biformer_stl_nchw import nchwBiFormerSTL
from timm.models.vision_transformer import _cfg

# Instantiate the models
convnext_model = ConvBraNet()
biformer_model = nchwBiFormerSTL()

class ConvNeXtBiFormerFusion(nn.Module):
    def __init__(self, convnext_model, biformer_model, num_classes, dims=[40, 80, 160, 320]):
        super(ConvNeXtBiFormerFusion, self).__init__()
        self.convnext_downsample_layers = convnext_model.downsample_layers
        self.convnext_stages = convnext_model.stages
        self.biformer_downsample_layers = biformer_model.downsample_layers
        self.biformer_stages = biformer_model.stages
        self.num_classes = num_classes
        # You can try different fusion strategies
        self.fusion_weight = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        # # Add a fully connected layer
        # self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        convnext_features = []
        biformer_features = []

        initial_x = x  # Save the initial input value

        # ConvNeXt loop
        for i in range(4):
            x = self.convnext_downsample_layers[i](x)
            x = self.convnext_stages[i](x)
            convnext_features.append(x)

        # Use the initial input value for BiFormer loop
        x = initial_x  # Restore the initial input value
        for i in range(4):
            x = self.biformer_downsample_layers[i](x)
            x = self.biformer_stages[i](x)
            biformer_features.append(x)

        # Get the output of the last stage from both models
        convnext_output = convnext_features[-1]
        biformer_output = biformer_features[-1]

        # Perform weighted fusion of the outputs from both models
        fused_output = self.fusion_weight * convnext_output + (1 - self.fusion_weight) * biformer_output
        fused_output = self.norm(fused_output.mean([-2, -1]))
        # print(fused_output.shape)
        classification_scores = self.head(fused_output)
        return classification_scores

def ConvNeXtBiFormerFusionatto(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    # Instantiate the models
    convnext_model = ConvBraNet()
    biformer_model = nchwBiFormerSTL()

    model = ConvNeXtBiFormerFusion(
        convnext_model=convnext_model,
        biformer_model=biformer_model,
        # in_chans=3, num_classes=1000,
        # depth=[2, 2, 6, 2],
        # embed_dim=[40, 80, 160, 320],
        # head_dim=32, qk_scale=None,
        # drop_path_rate=0., drop_rate=0.,
        # use_checkpoint_stages=[],
        # # before_attn_dwconv=3,
        # mlp_ratios=[4, 4, 4, 4],
        # norm_layer=LayerNorm2d,
        # pre_head_norm_layer=None,
        # ######## biformer specific ############
        # n_wins: Union[int, Tuple[int]] = (7, 7, 7, 7),
        #                                  topks: Union[int, Tuple[int]] = (1, 4, 16, -1),
        #                                                                  side_dwconv: int = 5,
        **kwargs)
    model.default_cfg = _cfg()

    return model

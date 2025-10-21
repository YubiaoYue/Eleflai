import torch.nn as nn
import torch
from timm.models.registry import register_model
from typing import List
from ultralytics import YOLO


class YourConvNet(nn.Module):
    def __init__(self, model_backbone=None):
        super().__init__()

        if model_backbone is None:
            raise ValueError("model_backbone cannot be None")

        # 只提取 backbone 的各个阶段
        self.layer1 = nn.Sequential(model_backbone.model.model[0])
        self.layer2 = nn.Sequential(model_backbone.model.model[1],
                                    model_backbone.model.model[2])
        self.layer3 = nn.Sequential(model_backbone.model.model[3],
                                    model_backbone.model.model[4])
        self.layer4 = nn.Sequential(model_backbone.model.model[5])
        self.layer5 = nn.Sequential(*[model_backbone.model.model[i]
                                      for i in range(6, 10)])

    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 32

    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return [128, 240, 360, 480]

    def forward(self, x, hierarchical=False):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            hierarchical:
                - True: 返回多尺度特征（用于预训练）
                - False: 返回最后一层特征（简化版）
        """
        x = self.layer1(x)

        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)

        if hierarchical:
            # 预训练时返回多尺度特征
            return [feat2, feat3, feat4, feat5]
        else:
            # 简单返回最后一层（如果框架要求）
            return feat5


@register_model
def your_convnet(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet.")
    return YourConvNet(YOLO("yolov9m.yaml"))


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('your_convnet')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())

    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
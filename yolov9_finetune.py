import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics import YOLO
from collections import OrderedDict


def remove_module_prefix(state_dict):
    """
    移除 state_dict 中的 'module.' 前缀（多GPU训练产生的）
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # 移除 'module.' 前缀
            name = k[7:]  # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def add_module_prefix(state_dict):
    """
    为 state_dict 添加 'module.' 前缀
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            name = 'module.' + k
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def smart_load_state_dict(model, state_dict, strict=True):
    """
    智能加载 state_dict，自动处理多GPU训练的权重
    """
    # 获取模型的第一个键和state_dict的第一个键
    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())

    if len(model_keys) == 0 or len(state_keys) == 0:
        model.load_state_dict(state_dict, strict=strict)
        return

    model_has_module = model_keys[0].startswith('module.')
    state_has_module = state_keys[0].startswith('module.')

    # 情况1: 模型没有'module.'前缀，但state_dict有 -> 移除前缀
    if not model_has_module and state_has_module:
        print("⚠️  检测到多GPU训练的权重，正在移除 'module.' 前缀...")
        state_dict = remove_module_prefix(state_dict)

    # 情况2: 模型有'module.'前缀，但state_dict没有 -> 添加前缀
    elif model_has_module and not state_has_module:
        print("⚠️  模型使用多GPU，正在添加 'module.' 前缀...")
        state_dict = add_module_prefix(state_dict)

    # 情况3: 都有或都没有 -> 直接加载
    else:
        print("✓ 权重格式匹配，直接加载")

    model.load_state_dict(state_dict, strict=strict)


# 1. 加载 YOLO
yolo = YOLO("yolov9m.yaml")
backbone = yolo.model.model  # nn.Sequential


# 2. 定义与上次完全相同结构的 PretrainNet
class PretrainNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stage1 = deepcopy(nn.Sequential(backbone[0]))  # 0
        self.stage2 = deepcopy(nn.Sequential(backbone[1], backbone[2]))  # 1:3
        self.stage3 = deepcopy(nn.Sequential(backbone[3], backbone[4]))  # 3:5
        self.stage4 = deepcopy(nn.Sequential(backbone[5]))  # 5:6
        self.stage5 = deepcopy(nn.Sequential(*[backbone[i] for i in range(6, 10)]))  # 6:10

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


# 3. 创建空的 PretrainNet 并加载本地权重（支持多GPU训练的权重）
pretrain_net = PretrainNet(backbone)
state_dict = torch.load("pretrain_net_v9.pth", map_location="cpu")

# 处理可能的多GPU checkpoint格式
if 'model' in state_dict:
    # 有些训练框架会把model包在字典里
    state_dict = state_dict['model']
elif 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

# 智能加载权重
smart_load_state_dict(pretrain_net, state_dict, strict=True)
print("✅ Loaded pretrain_net.pth successfully.")

# 4. 把预训练参数迁移到 YOLO backbone
stage1_yolo = nn.Sequential(backbone[0])
stage2_yolo = nn.Sequential(backbone[1], backbone[2])
stage3_yolo = nn.Sequential(backbone[3], backbone[4])
stage4_yolo = nn.Sequential(backbone[5])
stage5_yolo = nn.Sequential(*[backbone[i] for i in range(6, 10)])

with torch.no_grad():
    stage1_yolo.load_state_dict(pretrain_net.stage1.state_dict(), strict=True)
    stage2_yolo.load_state_dict(pretrain_net.stage2.state_dict(), strict=True)
    stage3_yolo.load_state_dict(pretrain_net.stage3.state_dict(), strict=True)
    stage4_yolo.load_state_dict(pretrain_net.stage4.state_dict(), strict=True)
    stage5_yolo.load_state_dict(pretrain_net.stage5.state_dict(), strict=True)

print("✅ Transferred pretrained weights back into YOLO backbone.")

# Train the model
results = yolo.train(data="OTO160K.yaml", epochs=50, imgsz=640, freeze=10, optimizer="auto", amp=True,
                     lr0=0.01, lrf=0.01,momentum=0.937,weight_decay=0.0005,warmup_epochs=3.0,warmup_momentum=0.8,
                     warmup_bias_lr=0.1)
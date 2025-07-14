import torch
from yacs.config import CfgNode as CN
from layers.cls_cvt import get_cls_model
import os

# 创建配置对象
config = CN()
config.MODEL = CN()
config.MODEL.SPEC = CN()

# 填充配置信息
config.MODEL.SPEC.INIT = 'trunc_norm'
config.MODEL.SPEC.NUM_STAGES = 3
config.MODEL.SPEC.PATCH_SIZE = [7, 3, 3]
config.MODEL.SPEC.PATCH_STRIDE = [4, 2, 2]
config.MODEL.SPEC.PATCH_PADDING = [2, 1, 1]
config.MODEL.SPEC.DIM_EMBED = [64, 192, 384]
config.MODEL.SPEC.NUM_HEADS = [1, 3, 6]
config.MODEL.SPEC.DEPTH = [1, 2, 10]
config.MODEL.SPEC.MLP_RATIO = [4.0, 4.0, 4.0]
config.MODEL.SPEC.ATTN_DROP_RATE = [0.0, 0.0, 0.0]
config.MODEL.SPEC.DROP_RATE = [0.0, 0.0, 0.0]
config.MODEL.SPEC.DROP_PATH_RATE = [0.0, 0.0, 0.1]
config.MODEL.SPEC.QKV_BIAS = [True, True, True]
config.MODEL.SPEC.CLS_TOKEN = [False, False, False]
config.MODEL.SPEC.POS_EMBED = [False, False, False]
config.MODEL.SPEC.QKV_PROJ_METHOD = ['dw_bn', 'dw_bn', 'dw_bn']
config.MODEL.SPEC.KERNEL_QKV = [3, 3, 3]
config.MODEL.SPEC.PADDING_KV = [1, 1, 1]
config.MODEL.SPEC.STRIDE_KV = [2, 2, 2]
config.MODEL.SPEC.PADDING_Q = [1, 1, 1]
config.MODEL.SPEC.STRIDE_Q = [1, 1, 1]
config.MODEL.NUM_CLASSES = 10  # 假设类别数为 10
config.MODEL.PRETRAINED = ''
config.MODEL.PRETRAINED_LAYERS = ['*']
config.MODEL.INIT_WEIGHTS = True
pretrained='CvT-13-384x384-IN-22k.pth'
config.VERBOSE = True

current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
pretrained_path = os.path.join(current_dir, '../../pretrained_models')
config.MODEL.PRETRAINED = os.path.join(pretrained_path, pretrained)
# 创建模型实例
model = get_cls_model(config)

# 生成随机输入数据
batch_size = 2
input_channels = 3
height = 256
width = 256
x = torch.randn(batch_size, input_channels, height, width)
z = torch.randn(batch_size, input_channels, height, width)

# 进行前向传播
output = model(z,x)

# 打印输出结果的形状
print(f"Output shape: {output.shape}")
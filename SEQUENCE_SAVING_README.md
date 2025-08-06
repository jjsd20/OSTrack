# 序列保存功能说明

这个功能允许你将数据加载器（sampler）生成的序列数据保存到文件中，方便后续分析和使用。

## 功能特点

- **支持多种保存格式**: JSON 和 Pickle
- **灵活的图像保存**: 可选择是否保存图像文件
- **完整的元数据**: 保存序列的所有相关信息
- **批量保存**: 支持批量保存多个序列
- **易于加载**: 提供简单的数据加载接口

## 文件结构

```
saved_sequences/
├── seq_000001_20241201_143022/
│   ├── metadata.json          # 序列元数据
│   ├── annotations.json       # 标注数据
│   └── images/                # 图像文件（可选）
│       ├── template/
│       │   ├── template_000.jpg
│       │   └── template_001.jpg
│       └── search/
│           ├── search_000.jpg
│           └── search_001.jpg
├── seq_000002_20241201_143025/
│   ├── metadata.json
│   ├── annotations.json
│   └── images/
└── saved_paths_20241201_143030.txt  # 所有保存路径的列表
```

## 使用方法

### 1. 基本使用

```python
from lib.train.data.sequence_saver import SequenceSaver

# 创建保存器
saver = SequenceSaver("my_sequences")

# 保存单个序列
data = sampler[0]  # 从采样器获取数据
save_path = saver.save_sequence_data(data, "seq_001", save_images=True)
```

### 2. 批量保存

```python
from torch.utils.data import DataLoader

# 创建数据加载器
dataloader = DataLoader(sampler, batch_size=1, shuffle=False, num_workers=0)

# 批量保存
saved_paths = saver.save_batch_sequences(dataloader, num_sequences=100, save_images=True)
```

### 3. 加载保存的数据

```python
# 加载序列数据
metadata, annotations = saver.load_sequence_data(save_path)

print("元数据:", metadata)
print("标注数据:", annotations.keys())
```

## 保存的数据内容

### 元数据 (metadata.json)
```json
{
  "sequence_id": "seq_001",
  "timestamp": "20241201_143022",
  "dataset": "lasot",
  "test_class": "person",
  "is_video_dataset": true,
  "template_frame_ids": [10, 11, 12],
  "search_frame_ids": [15, 16],
  "gt_frame_ids_backward": [10, 11, 12, 13, 14],
  "gt_frame_ids_forward": [15, 16, 17, 18, 19]
}
```

### 标注数据 (annotations.json)
```json
{
  "template_anno": [[100, 100, 50, 50]],
  "search_anno": [[120, 120, 60, 60]],
  "gt_sequence_anno_backward": [[...], [...], ...],
  "gt_sequence_anno_forward": [[...], [...], ...]
}
```

## 脚本使用

### 1. 环境检查（推荐先运行）
```bash
python check_environment.py
```

### 2. 运行演示脚本
```bash
python save_sequences_simple.py
```

### 3. 运行真实数据保存脚本
```bash
python save_sequences_with_real_data.py
```

### 4. 运行完整示例
```bash
python save_sequences_example.py
```

## 自定义使用

### 1. 修改采样器创建

在 `save_sequences_simple.py` 中，你需要创建你的采样器：

```python
from lib.train.data.sampler import TimingTrackingSampler
from lib.train.data.processing import Compose, ToTensor, Normalize

# 创建处理管道
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建采样器
sampler = TimingTrackingSampler(
    datasets=your_datasets,
    p_datasets=your_p_datasets,
    samples_per_epoch=1000,
    max_gap=200,
    num_search_frames=1,
    num_template_frames=1,
    processing=transform,
    frame_sample_mode='causal'
)

# 保存序列
save_sequences_from_sampler(sampler, num_sequences=100, save_dir="my_sequences")
```

### 2. 保存格式选择

```python
# 保存为JSON格式（默认，人类可读）
saver.save_sequence_data(data, "seq_001", save_format="json")

# 保存为Pickle格式（更紧凑，加载更快）
saver.save_sequence_data(data, "seq_001", save_format="pickle")
```

### 3. 图像保存控制

```python
# 保存图像（默认）
saver.save_sequence_data(data, "seq_001", save_images=True)

# 不保存图像（节省空间）
saver.save_sequence_data(data, "seq_001", save_images=False)
```

## 注意事项

1. **存储空间**: 保存图像会占用大量存储空间，建议根据需要选择
2. **数据格式**: 确保你的数据格式与保存器期望的格式一致
3. **路径管理**: 保存路径会自动创建，但确保有足够的磁盘空间
4. **错误处理**: 保存过程中如果某个序列出错，会跳过并继续处理下一个

## 故障排除

### 常见问题

1. **导入错误**: 确保 `lib` 目录在 Python 路径中
2. **数据格式错误**: 检查数据字典的键名是否正确
3. **存储空间不足**: 检查磁盘空间，特别是保存图像时
4. **权限错误**: 确保对保存目录有写权限
5. **数据集路径错误**: 检查 `lib/train/admin/local.py` 中的数据集路径设置
6. **配置文件缺失**: 确保配置文件存在且格式正确

### 环境检查

运行环境检查脚本来自动诊断问题：

```bash
python check_environment.py
```

这个脚本会检查：
- 环境设置文件 (`local.py`)
- 数据集路径是否存在
- 配置文件是否存在
- 模块导入是否成功

### 数据集路径设置

如果数据集路径未设置，请编辑 `lib/train/admin/local.py` 文件：

```python
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/path/to/your/workspace'
        self.lasot_dir = '/path/to/lasot/dataset'
        self.got10k_dir = '/path/to/got10k/dataset'
        self.coco_dir = '/path/to/coco/dataset'
        # ... 其他数据集路径
```

### 调试建议

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据格式
print("数据键:", data.keys())
print("数据类型:", {k: type(v) for k, v in data.items()})
```

## 扩展功能

你可以根据需要扩展 `SequenceSaver` 类：

- 添加更多数据格式支持
- 实现数据压缩
- 添加数据验证
- 支持增量保存
- 添加数据统计功能 
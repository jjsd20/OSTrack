#!/usr/bin/env python3
"""
环境检查脚本
检查数据集路径和配置是否正确设置
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.train.admin.environment import env_settings


def check_environment():
    """检查环境设置"""
    print("=== 环境设置检查 ===")
    
    try:
        env = env_settings()
        print("✓ 环境设置加载成功")
    except Exception as e:
        print(f"✗ 环境设置加载失败: {e}")
        print("请检查 lib/train/admin/local.py 文件是否存在且配置正确")
        return False
    
    # 检查工作目录
    print(f"\n工作目录: {env.workspace_dir}")
    if env.workspace_dir and os.path.exists(env.workspace_dir):
        print("✓ 工作目录存在")
    else:
        print("✗ 工作目录不存在或未设置")
    
    # 检查数据集目录
    datasets_to_check = [
        ('LaSOT', env.lasot_dir),
        ('GOT-10k', env.got10k_dir),
        ('GOT-10k Val', env.got10k_val_dir),
        ('COCO', env.coco_dir),
        ('ImageNet VID', env.imagenet_dir),
        ('TrackingNet', env.trackingnet_dir),
    ]
    
    print("\n=== 数据集路径检查 ===")
    available_datasets = []
    
    for name, path in datasets_to_check:
        if path and os.path.exists(path):
            print(f"✓ {name}: {path}")
            available_datasets.append(name)
        else:
            print(f"✗ {name}: {path or '未设置'}")
    
    return len(available_datasets) > 0, available_datasets


def check_config_files():
    """检查配置文件"""
    print("\n=== 配置文件检查 ===")
    
    config_files = [
        "lib/config/ostrack/config.py",
        "lib/config/timostrack/config.py",
        "lib/config/cvtostrack/config.py",
    ]
    
    available_configs = []
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file}")
            available_configs.append(config_file)
        else:
            print(f"✗ {config_file}")
    
    return available_configs


def check_dataset_imports():
    """检查数据集导入"""
    print("\n=== 数据集模块检查 ===")
    
    try:
        from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
        print("✓ 数据集模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 数据集模块导入失败: {e}")
        return False


def check_sampler_imports():
    """检查采样器导入"""
    print("\n=== 采样器模块检查 ===")
    
    try:
        from lib.train.data import sampler, opencv_loader, processing
        import lib.train.data.transforms as tfm
        print("✓ 采样器模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 采样器模块导入失败: {e}")
        return False


def create_test_local_file():
    """创建测试用的local.py文件"""
    print("\n=== 创建测试local.py文件 ===")
    
    local_file_path = "lib/train/admin/local.py"
    
    if os.path.exists(local_file_path):
        print(f"local.py文件已存在: {local_file_path}")
        return
    
    # 创建测试配置
    test_config = '''class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/tmp/ostrack_workspace'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/tmp/ostrack_workspace/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/tmp/ostrack_workspace/pretrained_networks'
        self.lasot_dir = ''    # LaSOT数据集路径
        self.got10k_dir = ''    # GOT-10k数据集路径
        self.got10k_val_dir = ''    # GOT-10k验证集路径
        self.trackingnet_dir = ''    # TrackingNet数据集路径
        self.coco_dir = ''    # COCO数据集路径
        self.imagenet_dir = ''    # ImageNet VID数据集路径
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        # LMDB格式数据集路径
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_lmdb_dir = ''
        self.imagenet_lmdb_dir = ''
'''
    
    try:
        with open(local_file_path, 'w') as f:
            f.write(test_config)
        print(f"✓ 已创建测试local.py文件: {local_file_path}")
        print("请根据你的实际数据集路径修改此文件")
    except Exception as e:
        print(f"✗ 创建local.py文件失败: {e}")


def main():
    """主函数"""
    print("OSTrack环境检查工具")
    print("=" * 50)
    
    # 检查环境设置
    env_ok, available_datasets = check_environment()
    
    # 检查配置文件
    available_configs = check_config_files()
    
    # 检查模块导入
    dataset_import_ok = check_dataset_imports()
    sampler_import_ok = check_sampler_imports()
    
    # 总结
    print("\n=== 检查总结 ===")
    
    if env_ok and dataset_import_ok and sampler_import_ok:
        print("✓ 环境配置基本正确")
        if available_datasets:
            print(f"✓ 可用数据集: {', '.join(available_datasets)}")
        else:
            print("⚠ 没有可用的数据集，将使用模拟数据")
        
        if available_configs:
            print(f"✓ 可用配置文件: {len(available_configs)} 个")
        else:
            print("⚠ 没有找到配置文件")
        
        print("\n可以运行序列保存脚本了！")
        
    else:
        print("✗ 环境配置有问题")
        
        if not env_ok:
            print("- 环境设置文件有问题")
            create_test_local_file()
        
        if not dataset_import_ok:
            print("- 数据集模块导入失败")
        
        if not sampler_import_ok:
            print("- 采样器模块导入失败")
        
        print("\n请修复上述问题后再运行序列保存脚本")


if __name__ == "__main__":
    main() 
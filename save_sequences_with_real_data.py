#!/usr/bin/env python3
"""
使用真实数据集配置的序列保存脚本
参考训练脚本的数据集创建方式，支持加载YAML配置文件
"""

import sys
import os
import torch
import importlib
import yaml
from easydict import EasyDict as edict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.train.data.sequence_saver import SequenceSaver
from lib.train.data import sampler, opencv_loader, processing
import lib.train.data.transforms as tfm
from lib.train.admin.environment import env_settings


def load_yaml_config(config_path):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 转换为EasyDict
    config = edict(config_dict)
    print(f"成功加载配置文件: {config_path}")
    return config


def create_settings_and_config(config_file=None):
    """创建设置和配置，参考训练脚本"""
    
    # 创建基本设置
    class Settings:
        def __init__(self):
            self.script_name = "timostrack"  # 或者 "ostrack"
            self.local_rank = -1
            self.use_lmdb = False  # 是否使用LMDB格式
    
    settings = Settings()
    
    # 加载环境设置
    try:
        settings.env = env_settings()
    except RuntimeError as e:
        print(f"环境设置错误: {e}")
        print("请确保已正确设置 lib/train/admin/local.py 文件")
        return None, None
    
    # 加载配置文件
    # if config_file and os.path.exists(config_file):
    #     # 加载YAML配置文件
    #     cfg = load_yaml_config(config_file)
    #     if cfg is None:
    #         print("无法加载YAML配置文件，使用默认配置")
    #         cfg = create_default_config()
    # else:
    #     # 尝试加载默认配置文件
    #     try:
    #         config_module = importlib.import_module("lib.config.timostrack.config")
    #         cfg = config_module.cfg
    #         print("成功加载默认配置文件")
    #     except ImportError:
    #         print("无法加载默认配置文件，使用默认配置")
    #         cfg = create_default_config()

    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(config_file)
    
    return settings, cfg


def create_default_config():
    """创建默认配置"""
    class DefaultConfig:
        def __init__(self):
            self.DATA = type('obj', (object,), {
                'TRAIN': type('obj', (object,), {
                    'DATASETS_NAME': ['LASOT'],
                    'DATASETS_RATIO': [1.0],
                    'SAMPLE_PER_EPOCH': 1000
                }),
                'VAL': type('obj', (object,), {
                    'DATASETS_NAME': ['LASOT'],
                    'DATASETS_RATIO': [1.0],
                    'SAMPLE_PER_EPOCH': 100
                }),
                'MAX_SAMPLE_INTERVAL': 200,
                'TEMPLATE': type('obj', (object,), {
                    'SIZE': 128,
                    'FACTOR': 2.0,
                    'CENTER_JITTER': 0.0,
                    'SCALE_JITTER': 0.0,
                    'NUMBER': 1
                }),
                'SEARCH': type('obj', (object,), {
                    'SIZE': 256,
                    'FACTOR': 4.0,
                    'CENTER_JITTER': 0.0,
                    'SCALE_JITTER': 0.0,
                    'NUMBER': 1
                }),
                'SAMPLER_MODE': 'causal',
                'MEAN': [0.485, 0.456, 0.406],
                'STD': [0.229, 0.224, 0.225]
            })
            self.TRAIN = type('obj', (object,), {
                'BATCH_SIZE': 1,
                'NUM_WORKER': 0,
                'TRAIN_CLS': False,
                'VAL_EPOCH_INTERVAL': 1
            })
            self.TIMING = type('obj', (object,), {
                'seq_len': 50,
                'pred_len': 1,
                'd_model': 16,
                'embed': 'timeF',
                'freq': 'h',
                'dropout': 0.1,
                'e_layers': 3,
                'top_k': 3,
                'num_kernels': 6,
                'd_ff': 32
            })
    
    return DefaultConfig()


def create_datasets(settings, cfg):
    """创建数据集，参考 base_functions.py"""
    
    def names2datasets(name_list, settings, image_loader):
        """数据集名称到数据集对象的转换"""
        from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
        
        datasets = []
        for name in name_list:
            if name == "LASOT":
                if hasattr(settings.env, 'lasot_dir') and settings.env.lasot_dir:
                    print(f"Building LaSOT dataset from: {settings.env.lasot_dir}")
                    datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
                else:
                    print("LaSOT数据集路径未设置，跳过")
            elif name == "GOT10K_vottrain":
                if hasattr(settings.env, 'got10k_dir') and settings.env.got10k_dir:
                    print(f"Building GOT-10k dataset from: {settings.env.got10k_dir}")
                    datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
                else:
                    print("GOT-10k数据集路径未设置，跳过")
            elif name == "GOT10K_votval":
                if hasattr(settings.env, 'got10k_val_dir') and settings.env.got10k_val_dir:
                    print(f"Building GOT-10k validation dataset from: {settings.env.got10k_val_dir}")
                    datasets.append(Got10k(settings.env.got10k_val_dir, split='votval', image_loader=image_loader))
                else:
                    print("GOT-10k验证数据集路径未设置，跳过")
            elif name == "COCO17":
                if hasattr(settings.env, 'coco_dir') and settings.env.coco_dir:
                    print(f"Building COCO dataset from: {settings.env.coco_dir}")
                    datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
                else:
                    print("COCO数据集路径未设置，跳过")
            elif name == "TRACKINGNET":
                if hasattr(settings.env, 'trackingnet_dir') and settings.env.trackingnet_dir:
                    print(f"Building TrackingNet dataset from: {settings.env.trackingnet_dir}")
                    datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
                else:
                    print("TrackingNet数据集路径未设置，跳过")
            else:
                print(f"不支持的数据集: {name}")
        
        return datasets
    
    # 创建数据集
    datasets = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader)
    
    if not datasets:
        print("没有可用的数据集，创建模拟数据集")
        return None
    
    return datasets


def create_sampler(settings, cfg, datasets):
    """创建采样器"""
    
    # 数据变换
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))
    
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    
    # 处理模块
    output_sz = {'template': cfg.DATA.TEMPLATE.SIZE, 'search': cfg.DATA.SEARCH.SIZE}
    search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR, 'search': cfg.DATA.SEARCH.FACTOR}
    center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER, 'search': cfg.DATA.SEARCH.CENTER_JITTER}
    scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER, 'search': cfg.DATA.SEARCH.SCALE_JITTER}
    
    data_processing = processing.STARKProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=center_jitter_factor,
        scale_jitter_factor=scale_jitter_factor,
        mode='sequence',
        transform=transform_train,
        joint_transform=transform_joint,
        settings=settings
    )
    
    # 创建采样器
    if settings.script_name == "timostrack":
        sampler_obj = sampler.TimingTrackingSampler(
            datasets=datasets,
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            num_search_frames=cfg.DATA.SEARCH.NUMBER,
            num_template_frames=cfg.DATA.TEMPLATE.NUMBER,
            processing=data_processing,
            frame_sample_mode=getattr(cfg.DATA, 'SAMPLER_MODE', 'causal'),
            train_cls=getattr(cfg.TRAIN, 'TRAIN_CLS', False)
        )
    else:
        sampler_obj = sampler.TrackingSampler(
            datasets=datasets,
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            num_search_frames=cfg.DATA.SEARCH.NUMBER,
            num_template_frames=cfg.DATA.TEMPLATE.NUMBER,
            processing=data_processing,
            frame_sample_mode=getattr(cfg.DATA, 'SAMPLER_MODE', 'causal'),
            train_cls=getattr(cfg.TRAIN, 'TRAIN_CLS', False)
        )
    
    return sampler_obj


def save_sequences_with_real_data(num_sequences=10, save_dir="real_sequences", save_images=True, config_file=None):
    """使用真实数据保存序列"""
    
    print("=== 使用真实数据集保存序列 ===")
    
    # 创建设置和配置
    settings, cfg = create_settings_and_config(config_file)
    if settings is None:
        print("无法创建设置，退出")
        return
    
    # 打印配置信息
    print(f"配置信息:")
    print(f"  - 数据集: {cfg.DATA.TRAIN.DATASETS_NAME}")
    print(f"  - 样本数: {cfg.DATA.TRAIN.SAMPLE_PER_EPOCH}")
    print(f"  - 模板大小: {cfg.DATA.TEMPLATE.SIZE}")
    print(f"  - 搜索大小: {cfg.DATA.SEARCH.SIZE}")
    if hasattr(cfg, 'TIMING'):
        print(f"  - 序列长度: {cfg.TIMING.seq_len}")
        print(f"  - 预测长度: {cfg.TIMING.pred_len}")
    
    # 创建数据集
    datasets = create_datasets(settings, cfg)
    if datasets is None:
        print("无法创建数据集，使用模拟数据")
        save_sequences_with_mock_data(num_sequences, save_dir, save_images, cfg)
        return
    
    # 创建采样器
    sampler_obj = create_sampler(settings, cfg, datasets)
    
    # 创建保存器
    saver = SequenceSaver(save_dir)
    
    # 初始化计时变量（参考ltr_trainer.py）
    import time
    num_frames = 0
    start_time = time.time()
    prev_time = start_time
    avg_data_time = 0
    total_data_time = 0
    
    # 保存序列
    saved_paths = []
    
    print(f"\n开始数据加载性能测试...")
    print(f"{'序列':<6} {'单次时间(s)':<12} {'累计平均(s)':<12} {'FPS':<8}")
    print("-" * 45)
    
    for i in range(num_sequences):
        try:
            # 记录数据读取开始时间
            data_start_time = time.time()
            
            # 从采样器获取数据（这里是主要的数据加载时间）
            data = sampler_obj[i]
            
            # 记录数据读取完成时间
            data_read_done_time = time.time()
            
            # 计算本次数据加载时间
            current_data_time = data_read_done_time - data_start_time
            total_data_time += current_data_time
            num_frames += 1
            
            # 计算平均数据加载时间和FPS
            avg_data_time = total_data_time / num_frames
            current_fps = 1.0 / current_data_time if current_data_time > 0 else 0
            avg_fps = num_frames / (data_read_done_time - start_time)
            
            # 保存序列
            save_path = saver.save_sequence_data(
                data, 
                f"real_seq_{i:06d}", 
                save_images=save_images,
                save_format="json"
            )
            saved_paths.append(save_path)
            
            # 打印计时信息（每5个序列或最后一个）
            if (i + 1) % 5 == 0 or i == num_sequences - 1:
                print(f"{i+1:<6} {current_data_time:<12.3f} {avg_data_time:<12.3f} {avg_fps:<8.1f}")
                
        except Exception as e:
            print(f"保存序列 {i} 时出错: {e}")
            continue
    
    # 打印最终统计信息
    total_time = time.time() - start_time
    print("-" * 45)
    print(f"数据加载性能统计:")
    print(f"  - 总序列数: {num_frames}")
    print(f"  - 总时间: {total_time:.3f}s")
    print(f"  - 平均数据加载时间: {avg_data_time:.3f}s")
    print(f"  - 平均FPS: {num_frames / total_time:.1f}")
    print(f"  - 最快单次: {min([total_data_time / num_frames] * num_frames) if num_frames > 0 else 0:.3f}s")
    print(f"  - 数据加载占比: {(total_data_time / total_time * 100):.1f}%")
    
    print(f"\n保存完成！共保存 {len(saved_paths)} 个序列")
    return saved_paths


def save_sequences_with_mock_data(num_sequences=10, save_dir="mock_sequences", save_images=True, cfg=None):
    """使用模拟数据保存序列"""
    
    print("=== 使用模拟数据保存序列 ===")
    
    # 创建保存器
    saver = SequenceSaver(save_dir)
    
    # 获取配置参数
    template_size = getattr(cfg.DATA.TEMPLATE, 'SIZE', 128) if cfg else 128
    search_size = getattr(cfg.DATA.SEARCH, 'SIZE', 256) if cfg else 256
    seq_len = getattr(cfg.TIMING, 'seq_len', 50) if cfg and hasattr(cfg, 'TIMING') else 50
    pred_len = getattr(cfg.TIMING, 'pred_len', 1) if cfg and hasattr(cfg, 'TIMING') else 1
    
    # 模拟序列数据
    for i in range(num_sequences):
        sample_data = {
            "template_images": torch.randn(1, 3, template_size, template_size),
            "search_images": torch.randn(1, 3, search_size, search_size),
            "template_anno": torch.tensor([[100 + i*10, 100 + i*10, 50, 50]]),
            "search_anno": torch.tensor([[120 + i*10, 120 + i*10, 60, 60]]),
            "gt_sequence_anno_backward": torch.randn(1, seq_len, 4),
            "gt_sequence_anno_forward": torch.randn(1, pred_len, 4),
            "dataset": "mock_dataset",
            "test_class": "mock_class",
            "is_video_dataset": True,
            "template_frame_ids": [10 + i, 11 + i, 12 + i],
            "search_frame_ids": [15 + i, 16 + i],
            "gt_frame_ids_backward": list(range(10 + i, 10 + i + seq_len)),
            "gt_frame_ids_forward": list(range(15 + i, 15 + i + pred_len)),
        }
        
        # 保存序列
        save_path = saver.save_sequence_data(sample_data, f"mock_{i:03d}", save_images=save_images)
        print(f"保存模拟序列 {i}: {save_path}")


def benchmark_data_loading(num_sequences=50, config_file=None):
    """专门的数据加载性能测试函数"""
    
    print("=== 数据加载性能基准测试 ===")
    
    # 创建设置和配置
    settings, cfg = create_settings_and_config(config_file)
    if settings is None:
        print("无法创建设置，退出")
        return
    
    # 创建数据集
    datasets = create_datasets(settings, cfg)
    if datasets is None:
        print("无法创建数据集，退出")
        return
    
    # 创建采样器
    sampler_obj = create_sampler(settings, cfg, datasets)
    
    # 初始化计时变量
    import time
    times = []
    
    print(f"开始测试 {num_sequences} 个序列的数据加载时间...")
    print(f"配置: {cfg.DATA.TRAIN.DATASETS_NAME}, 模板:{cfg.DATA.TEMPLATE.SIZE}, 搜索:{cfg.DATA.SEARCH.SIZE}")
    if hasattr(cfg, 'TIMING'):
        print(f"序列长度: {cfg.TIMING.seq_len}, 预测长度: {cfg.TIMING.pred_len}")
    print("-" * 60)
    
    # 预热（前几次可能较慢）
    print("预热中...")
    for i in range(3):
        try:
            _ = sampler_obj[i]
        except:
            pass
    
    print("开始正式测试...")
    start_total = time.time()
    
    for i in range(num_sequences):
        try:
            start_time = time.time()
            data = sampler_obj[i]
            end_time = time.time()
            
            load_time = end_time - start_time
            times.append(load_time)
            
            if (i + 1) % 10 == 0:
                avg_time = sum(times) / len(times)
                print(f"已测试 {i+1:3d}/{num_sequences}, 当前: {load_time:.3f}s, 平均: {avg_time:.3f}s, FPS: {1/avg_time:.1f}")
                
        except Exception as e:
            print(f"序列 {i} 加载失败: {e}")
            continue
    
    total_time = time.time() - start_total
    
    # 统计结果
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time
        
        print("\n" + "=" * 60)
        print("数据加载性能测试结果:")
        print(f"  - 成功测试序列数: {len(times)}")
        print(f"  - 平均加载时间: {avg_time:.3f}s")
        print(f"  - 最快加载时间: {min_time:.3f}s")
        print(f"  - 最慢加载时间: {max_time:.3f}s")
        print(f"  - 平均FPS: {fps:.1f}")
        print(f"  - 总测试时间: {total_time:.3f}s")
        print(f"  - 数据加载效率: {(sum(times)/total_time*100):.1f}%")
        
        # 分析数据组成
        if hasattr(data, 'keys'):
            print(f"\n数据组成分析:")
            for key in data.keys():
                if hasattr(data[key], 'shape'):
                    print(f"  - {key}: {data[key].shape}")
                elif hasattr(data[key], '__len__'):
                    print(f"  - {key}: length={len(data[key])}")
                else:
                    print(f"  - {key}: {type(data[key])}")
    else:
        print("没有成功的测试数据")


def test_data_loading_performance():
    """专门用于测试数据加载性能的函数"""
    
    # ========== 测试参数设置 ==========
    config_file = 'experiments/timostrack/vitb_256_mae_ce_96x1_ep300.yaml'
    test_sequences = 50  # 测试序列数量
    
    print("=" * 60)
    print("数据加载性能测试")
    print("=" * 60)
    print(f"配置文件: {config_file}")
    print(f"测试序列数: {test_sequences}")
    print("=" * 60)
    
    # 运行基准测试
    benchmark_data_loading(test_sequences, config_file)


def main():
    """主函数"""
    print("真实数据集序列保存工具")
    print("=" * 50)
    
    # ========== 直接在代码中设置参数 ==========
    config_file = 'experiments/timostrack/vitb_256_mae_ce_96x1_ep300.yaml'
    num_sequences = 50  # 要保存的序列数量
    save_dir = 'real_sequences'  # 保存目录
    save_images = False  # 是否保存图像文件
    check_env = True  # 是否检查环境设置
    
    # ========== 选择运行模式 ==========
    # 模式1: 纯性能测试（推荐用于测试优化效果）
    run_performance_test = False
    
    # 模式2: 保存序列并测试性能
    run_save_with_timing = True
    
    # 模式3: 只保存序列，不测试性能
    run_save_only = False
    
    print(f"配置参数:")
    print(f"  - 配置文件: {config_file}")
    print(f"  - 序列数量: {num_sequences}")
    print(f"  - 保存目录: {save_dir}")
    print(f"  - 保存图像: {save_images}")
    print(f"  - 性能测试模式: {run_performance_test}")
    print("-" * 50)
    
    # 检查环境设置
    if check_env:
        print("检查环境设置...")
        try:
            env = env_settings()
            print("✓ 环境设置加载成功")
            print(f"工作目录: {env.workspace_dir}")
            if hasattr(env, 'lasot_dir'):
                print(f"LaSOT目录: {env.lasot_dir}")
            if hasattr(env, 'got10k_dir'):
                print(f"GOT-10k目录: {env.got10k_dir}")
        except Exception as e:
            print(f"✗ 环境设置加载失败: {e}")
            print("将使用模拟数据")
        print("-" * 50)
    
    # 模式1: 纯性能测试
    if run_performance_test:
        print("开始运行数据加载性能基准测试...")
        benchmark_data_loading(num_sequences, config_file)
        return
    
    # 模式2: 保存序列并测试性能
    if run_save_with_timing:
        try:
            saved_paths = save_sequences_with_real_data(
                num_sequences=num_sequences, 
                save_dir=save_dir, 
                save_images=save_images,
                config_file=config_file
            )
            if saved_paths:
                print(f"成功保存 {len(saved_paths)} 个真实序列")
        except Exception as e:
            print(f"真实数据保存失败: {e}")
            print("使用模拟数据作为备选")
            save_sequences_with_mock_data(
                num_sequences=num_sequences, 
                save_dir=save_dir, 
                save_images=save_images
            )
        return
    
    # 模式3: 只保存序列
    if run_save_only:
        try:
            # 创建设置和配置
            settings, cfg = create_settings_and_config(config_file)
            if settings is None:
                print("无法创建设置，退出")
                return
            
            # 创建数据集
            datasets = create_datasets(settings, cfg)
            if datasets is None:
                print("无法创建数据集，使用模拟数据")
                save_sequences_with_mock_data(num_sequences, save_dir, save_images, cfg)
                return
            
            # 创建采样器
            sampler_obj = create_sampler(settings, cfg, datasets)
            
            # 创建保存器
            saver = SequenceSaver(save_dir)
            
            # 简单保存，不计时
            saved_paths = []
            for i in range(num_sequences):
                try:
                    data = sampler_obj[i]
                    save_path = saver.save_sequence_data(
                        data, 
                        f"seq_{i:06d}", 
                        save_images=save_images,
                        save_format="json"
                    )
                    saved_paths.append(save_path)
                    
                    if (i + 1) % 10 == 0:
                        print(f"已保存 {i + 1}/{num_sequences} 个序列")
                        
                except Exception as e:
                    print(f"保存序列 {i} 时出错: {e}")
                    continue
            
            print(f"保存完成！共保存 {len(saved_paths)} 个序列")
            
        except Exception as e:
            print(f"保存失败: {e}")
    
    print("\n" + "=" * 50)
    print("程序执行完成！")


if __name__ == "__main__":
    main() 
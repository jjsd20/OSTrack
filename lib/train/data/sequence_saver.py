import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from PIL import Image
import cv2


class SequenceSaver:
    """序列保存工具类，用于保存采样器生成的序列数据"""
    
    def __init__(self, save_dir="saved_sequences"):
        """
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建子目录
        self.images_dir = os.path.join(save_dir, "images")
        self.annotations_dir = os.path.join(save_dir, "annotations")
        self.metadata_dir = os.path.join(save_dir, "metadata")
        
        for dir_path in [self.images_dir, self.annotations_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_sequence_data(self, data, sequence_id, save_images=True, save_format="json"):
        """
        保存序列数据
        
        Args:
            data: 采样器返回的数据字典
            sequence_id: 序列ID
            save_images: 是否保存图像
            save_format: 保存格式 ("json" 或 "pickle")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seq_dir = os.path.join(self.save_dir, f"seq_{sequence_id}_{timestamp}")
        os.makedirs(seq_dir, exist_ok=True)
        
        # 保存元数据
        metadata = {
            "sequence_id": sequence_id,
            "timestamp": timestamp,
            "dataset": data.get("dataset", "unknown"),
            "test_class": data.get("test_class", "unknown"),
            "is_video_dataset": data.get("is_video_dataset", False),
            "template_frame_ids": data.get("template_frame_ids", []),
            "search_frame_ids": data.get("search_frame_ids", []),
            "gt_frame_ids_backward": data.get("gt_frame_ids_backward", []),
            "gt_frame_ids_forward": data.get("gt_frame_ids_forward", []),
        }
        
        if save_format == "json":
            self._save_json(metadata, os.path.join(seq_dir, "metadata.json"))
        else:
            self._save_pickle(metadata, os.path.join(seq_dir, "metadata.pkl"))
        
        # 保存标注数据
        annotations = {
            "template_anno": self._convert_to_list(data.get("template_anno", [])),
            "search_anno": self._convert_to_list(data.get("search_anno", [])),
            "gt_sequence_anno_backward": self._convert_to_list(data.get("gt_sequence_anno_backward", [])),
            "gt_sequence_anno_forward": self._convert_to_list(data.get("gt_sequence_anno_forward", [])),
        }
        
        if save_format == "json":
            self._save_json(annotations, os.path.join(seq_dir, "annotations.json"))
        else:
            self._save_pickle(annotations, os.path.join(seq_dir, "annotations.pkl"))
        
        # 保存图像
        if save_images:
            self._save_images(data, seq_dir)
        
        print(f"序列 {sequence_id} 已保存到 {seq_dir}")
        return seq_dir
    
    def save_batch_sequences(self, dataloader, num_sequences=100, save_images=True, save_format="json"):
        """
        批量保存序列数据
        
        Args:
            dataloader: 数据加载器
            num_sequences: 要保存的序列数量
            save_images: 是否保存图像
            save_format: 保存格式
        """
        saved_paths = []
        
        for i in range(num_sequences):
            try:
                data = next(iter(dataloader))
                if isinstance(data, list):
                    data = data[0]  # 如果是列表，取第一个元素
                
                save_path = self.save_sequence_data(data, i, save_images, save_format)
                saved_paths.append(save_path)
                
                if (i + 1) % 10 == 0:
                    print(f"已保存 {i + 1}/{num_sequences} 个序列")
                    
            except Exception as e:
                print(f"保存序列 {i} 时出错: {e}")
                continue
        
        # 保存保存路径列表
        paths_file = os.path.join(self.save_dir, f"saved_paths_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(paths_file, 'w') as f:
            for path in saved_paths:
                f.write(f"{path}\n")
        
        print(f"批量保存完成，共保存 {len(saved_paths)} 个序列")
        print(f"保存路径列表已保存到: {paths_file}")
        
        return saved_paths
    
    def _save_images(self, data, seq_dir):
        """保存图像数据"""
        images_dir = os.path.join(seq_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # 保存模板图像
        if "template_images" in data:
            template_dir = os.path.join(images_dir, "template")
            os.makedirs(template_dir, exist_ok=True)
            
            template_images = data["template_images"]
            if isinstance(template_images, torch.Tensor):
                template_images = template_images.cpu().numpy()
            
            for i, img in enumerate(template_images):
                if len(img.shape) == 3:
                    # 转换为RGB格式
                    if img.shape[0] == 3:  # CHW格式
                        img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(os.path.join(template_dir, f"template_{i:03d}.jpg"))
        
        # 保存搜索图像
        if "search_images" in data:
            search_dir = os.path.join(images_dir, "search")
            os.makedirs(search_dir, exist_ok=True)
            
            search_images = data["search_images"]
            if isinstance(search_images, torch.Tensor):
                search_images = search_images.cpu().numpy()
            
            for i, img in enumerate(search_images):
                if len(img.shape) == 3:
                    if img.shape[0] == 3:  # CHW格式
                        img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(os.path.join(search_dir, f"search_{i:03d}.jpg"))
    
    def _convert_to_list(self, data):
        """将张量转换为列表"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, list):
            return [self._convert_to_list(item) for item in data]
        else:
            return data
    
    def _save_json(self, data, filepath):
        """保存JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_pickle(self, data, filepath):
        """保存pickle文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_sequence_data(self, seq_dir, load_format="json"):
        """加载序列数据"""
        # 加载元数据
        if load_format == "json":
            with open(os.path.join(seq_dir, "metadata.json"), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            with open(os.path.join(seq_dir, "annotations.json"), 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        else:
            with open(os.path.join(seq_dir, "metadata.pkl"), 'rb') as f:
                metadata = pickle.load(f)
            with open(os.path.join(seq_dir, "annotations.pkl"), 'rb') as f:
                annotations = pickle.load(f)
        
        return metadata, annotations


def create_sequence_dataset_from_sampler(sampler, num_samples=1000, save_dir="sequence_dataset"):
    """
    从采样器创建序列数据集
    
    Args:
        sampler: 采样器实例
        num_samples: 样本数量
        save_dir: 保存目录
    """
    saver = SequenceSaver(save_dir)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(sampler, batch_size=1, shuffle=False, num_workers=0)
    
    # 批量保存
    saved_paths = saver.save_batch_sequences(dataloader, num_samples, save_images=True)
    
    return saved_paths


if __name__ == "__main__":
    # 使用示例
    saver = SequenceSaver("test_sequences")
    
    # 模拟数据
    test_data = {
        "template_images": torch.randn(1, 3, 256, 256),
        "search_images": torch.randn(1, 3, 256, 256),
        "template_anno": torch.tensor([[100, 100, 50, 50]]),
        "search_anno": torch.tensor([[120, 120, 60, 60]]),
        "gt_sequence_anno_backward": torch.randn(1, 50, 4),
        "gt_sequence_anno_forward": torch.randn(1, 5, 4),
        "dataset": "test_dataset",
        "test_class": "test_class",
        "is_video_dataset": True,
        "template_frame_ids": [10, 11, 12],
        "search_frame_ids": [15, 16],
        "gt_frame_ids_backward": list(range(10, 15)),
        "gt_frame_ids_forward": list(range(15, 20)),
    }
    
    # 保存单个序列
    save_path = saver.save_sequence_data(test_data, "test_001")
    print(f"保存路径: {save_path}")
    
    # 加载序列
    metadata, annotations = saver.load_sequence_data(save_path)
    print("加载的元数据:", metadata)
    print("加载的标注:", annotations.keys()) 
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
import numpy as np
from swift.llm import DatasetMeta, ResponsePreprocessor, MessagesPreprocessor, load_dataset, register_dataset
from swift.llm.dataset.loader import DatasetLoader


class CustomPreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/stsb',
        hf_dataset_id='SetFit/stsb',
        preprocess_func=CustomPreprocessor(),
    ))



class BDDPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # breakpoint()
        query = row.get('messages', '')[0]
        # breakpoint()
        assert query['role'] == 'user', 'this is not the user content!'
        # question_type = determine_question_type(query["content"])
        
        answer = row.get('messages', '')[1]
        assert answer['role'] =='assistant', 'this is not the assistant content!'
        
        # breakpoint()
        channel = 'public_bdd100k'
        row.update({'solutions': answer['content'], 'question_type': channel})
        # print(row)
        # breakpoint()
        return row
try:
    register_dataset(
        DatasetMeta(
            # ms_dataset_id='okwinds/clevr_cogen_a_train',
            # hf_dataset_id='ayeshaishaq/DriveLMMo1',
            dataset_path="/high_perf_store2/tfl-vepfs/zy/zy_utils/bdd100k_labels_images_train_weather_qwen_v1.json",
            preprocess_func=BDDPreprocessor(),
            # load_function=DatasetLoader.load(),
            tags=['qa', 'drive', 'vision', 'grpo']))
except:
    print(f'The `bdd-100k` has already been registered in the DATASET_MAPPING.')



class OmniDriveTrajectoryPreprocessor(MessagesPreprocessor):
    """
    处理OmniDrive轨迹数据的预处理器
    
    输入数据格式:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "... <image> ..."},  # 包含图片标记
            {"role": "assistant", "content": "高层级导航指令和解释"}
        ],
        "metadata": {
            "trajectory": [[x1, y1, h1], [x2, y2, h2], ...],  # 6个轨迹点
            "sample_idx": ...,
            "token": ...,
            "description": "..."
        },
        "images": ["path/to/image1.jpg", ...]
    }
    
    输出数据格式:
    {
        "messages": ... (保持不变),
        "metadata": ... (保持不变),
        "images": ... (保持不变),
        "solutions": "[0.1, 0.2, 0.3; 0.4, 0.5, 0.6; ...]"  # 格式化后的轨迹点
        "question_type": "omnidrive_trajectory"
    }
    """
    
    def __init__(self, 
                 trajectory_format: str = "compact",
                 normalize_coordinates: bool = False,
                 round_decimals: int = 6):
        """
        初始化预处理器
        
        Args:
            trajectory_format: 轨迹点格式化方式
                - "compact": 紧凑格式 "[x1,y1,h1; x2,y2,h2; ...]"
                - "list": 列表格式 "[[x1, y1, h1], [x2, y2, h2], ...]"
                - "detailed": 详细格式 "t=0.0s: x=0.1, y=0.2, h=0.3; ..."
            normalize_coordinates: 是否归一化坐标
            round_decimals: 小数点后保留位数
        """
        self.trajectory_format = trajectory_format
        self.normalize_coordinates = normalize_coordinates
        self.round_decimals = round_decimals
        super().__init__()
    
    def format_trajectory(self, trajectory: List[List[float]]) -> str:
        """
        格式化轨迹点
        
        Args:
            trajectory: 轨迹点列表，形状为 (6, 3) 或 (N, 3)
            
        Returns:
            格式化后的轨迹字符串
        """
        if not trajectory or len(trajectory) == 0:
            return ""
        
        # 确保轨迹是numpy数组
        traj_array = np.array(trajectory, dtype=np.float32)
        
        # 归一化处理（如果需要）
        if self.normalize_coordinates and len(traj_array) > 0:
            # 简单的归一化：减去第一个点
            traj_array = traj_array - traj_array[0]
        
        # 四舍五入
        traj_array = np.round(traj_array, self.round_decimals)
        
        if self.trajectory_format == "compact":
            # 紧凑格式: "[x1,y1,h1; x2,y2,h2; ...]"
            points = []
            for i, (x, y, h) in enumerate(traj_array):
                points.append(f"[{x:.{self.round_decimals}f},{y:.{self.round_decimals}f},{h:.{self.round_decimals}f}]")
            return "; ".join(points)
        
        elif self.trajectory_format == "list":
            # 列表格式: "[[x1, y1, h1], [x2, y2, h2], ...]"
            points = []
            for x, y, h in traj_array:
                points.append(f"[{x:.{self.round_decimals}f}, {y:.{self.round_decimals}f}, {h:.{self.round_decimals}f}]")
            return f"[{', '.join(points)}]"
        
        elif self.trajectory_format == "detailed":
            # 详细格式: "t=0.0s: x=0.1, y=0.2, h=0.3; ..."
            points = []
            for i, (x, y, h) in enumerate(traj_array):
                time = i * 0.5
                points.append(f"t={time:.1f}s: x={x:.{self.round_decimals}f}, y={y:.{self.round_decimals}f}, h={h:.{self.round_decimals}f}")
            return "; ".join(points)
        
        else:
            # 默认格式: 紧凑格式
            points = []
            for i, (x, y, h) in enumerate(traj_array):
                points.append(f"[{x:.{self.round_decimals}f},{y:.{self.round_decimals}f},{h:.{self.round_decimals}f}]")
            return "; ".join(points)
    
    def extract_navigation_instructions(self, messages: List[Dict[str, str]]) -> str:
        """
        从消息中提取高层级导航指令
        
        Args:
            messages: 消息列表
            
        Returns:
            导航指令文本
        """
        for msg in messages:
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
        return ""
    
    def extract_scene_description(self, metadata: Dict[str, Any]) -> str:
        """
        从metadata中提取场景描述
        
        Args:
            metadata: 元数据字典
            
        Returns:
            场景描述文本
        """
        description = metadata.get('description', '')
        if not description:
            location = metadata.get('location', '')
            scene_name = metadata.get('scene_name', '')
            if location or scene_name:
                return f"{location} - {scene_name}"
        return description
    
    def calculate_trajectory_statistics(self, trajectory: List[List[float]]) -> Dict[str, Any]:
        """
        计算轨迹统计数据
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            统计数据字典
        """
        if not trajectory:
            return {}
        
        traj_array = np.array(trajectory)
        
        # 计算位移
        start_point = traj_array[0, :2]  # (x, y)
        end_point = traj_array[-1, :2]
        displacement = np.sqrt(np.sum((end_point - start_point) ** 2))
        
        # 计算总距离
        distances = []
        for i in range(1, len(traj_array)):
            dist = np.sqrt(np.sum((traj_array[i, :2] - traj_array[i-1, :2]) ** 2))
            distances.append(dist)
        total_distance = sum(distances)
        
        # 计算平均速度 (距离/时间间隔)
        avg_speed = np.mean(distances) / 0.5 if distances else 0.0
        
        # 计算航向变化
        heading_change = traj_array[-1, 2] - traj_array[0, 2]
        
        return {
            "displacement": float(displacement),
            "total_distance": float(total_distance),
            "avg_speed": float(avg_speed),
            "heading_change": float(heading_change),
            "num_points": len(traj_array)
        }
    
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理函数
        
        Args:
            row: 原始数据行
            
        Returns:
            处理后的数据行
        """
        # 获取消息
        messages = row.get('messages', [])
        if not messages:
            raise ValueError("No messages found in row")
        
        # 验证消息格式
        user_found = False
        assistant_found = False
        for msg in messages:
            if msg.get('role') == 'user':
                user_found = True
            elif msg.get('role') == 'assistant':
                assistant_found = True
        
        if not user_found:
            raise ValueError("No user message found in messages")
        if not assistant_found:
            raise ValueError("No assistant message found in messages")
        
        # 获取metadata
        metadata = row.get('metadata', {})
        
        # 提取轨迹点
        trajectory = metadata.get('trajectory', [])
        if not trajectory:
            # 检查是否有其他可能的键
            for key in ['gt_planning', 'planning', 'traj']:
                if key in metadata:
                    trajectory = metadata[key]
                    if isinstance(trajectory, np.ndarray):
                        trajectory = trajectory.tolist()
                    break
        
        if not trajectory:
            raise ValueError(f"No trajectory data found in metadata. Available keys: {list(metadata.keys())}")
        
        # 确保轨迹是列表格式
        if isinstance(trajectory, np.ndarray):
            trajectory = trajectory.tolist()
        
        # 检查轨迹形状
        if len(trajectory) == 0:
            raise ValueError("Empty trajectory")
        
        # 如果轨迹是三维数组 (1, 6, 3)，展平成 (6, 3)
        if isinstance(trajectory[0], list) and isinstance(trajectory[0][0], list):
            # 可能是 (1, 6, 3) 的形状
            if len(trajectory) == 1 and len(trajectory[0]) == 6 and len(trajectory[0][0]) == 3:
                trajectory = trajectory[0]
            else:
                # 尝试展平
                import itertools
                trajectory = list(itertools.chain.from_iterable(trajectory))
        
        # 格式化轨迹点
        formatted_trajectory = self.format_trajectory(trajectory)
        
        # 提取导航指令
        navigation_instructions = self.extract_navigation_instructions(messages)
        
        # 提取场景描述
        scene_description = self.extract_scene_description(metadata)
        
        # 计算轨迹统计数据
        trajectory_stats = self.calculate_trajectory_statistics(trajectory)
        
        # 更新metadata
        metadata.update({
            'formatted_trajectory': formatted_trajectory,
            'navigation_instructions': navigation_instructions,
            'scene_description': scene_description,
            'trajectory_stats': trajectory_stats,
            'trajectory_original_length': len(trajectory),
            'trajectory_expected_length': 6  # 3秒轨迹，0.5秒间隔
        })
        
        # 更新行数据
        row.update({
            'trajectory': formatted_trajectory,  # 轨迹点作为解决方案
            'navigation': navigation_instructions,  # 导航指令
            'question_type': 'omnidrive_trajectory',
            'metadata': metadata
        })
        
        return row



# 注册数据集
try:
    # 注册轨迹数据集
    register_dataset(
        DatasetMeta(
            dataset_path="/path/to/omnidrive_trajectory_data.jsonl",  # 替换为实际路径
            preprocess_func=OmniDriveTrajectoryPreprocessor(trajectory_format="compact"),
            tags=['trajectory', 'planning', 'autonomous_driving', 'vla', 'ms-swift']
        )
    )
    print("Successfully registered OmniDrive trajectory dataset")
except Exception as e:
    print(f"Error registering OmniDrive trajectory dataset: {e}")

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

def get_system_prompt(prompt_type='default'):
    """
    获取系统提示
    """
    prompts = {
        'default': """You are an autonomous driving planner. Your task is to analyze the driving scene from the vehicle's cameras and generate a trajectory for the next 3 seconds with 6 waypoints, each spaced 0.5 seconds apart. Each waypoint should include x, y, and heading values relative to the current vehicle position.""",
        
        'detailed': """You are an expert autonomous driving planner. Your role is to:
1. Analyze the driving scene from multiple camera perspectives
2. Generate a safe and feasible trajectory for the next 3 seconds
3. Output 6 waypoints with x, y, and heading values (relative coordinates)
4. Each waypoint should be spaced 0.5 seconds apart
5. Consider traffic rules, obstacles, and road geometry

Please provide accurate trajectory predictions based on the visual information.""",
        
        'planning_focused': """You are a motion planning assistant. Given the current driving scene, generate a trajectory consisting of 6 waypoints (x, y, heading) for the next 3 seconds at 0.5-second intervals. Consider safety, comfort, and traffic rules in your planning."""
    }
    
    return prompts.get(prompt_type, prompts['default'])

def format_trajectory(waypoints, format_type='list'):
    """
    格式化轨迹点
    
    Args:
        waypoints: (6, 3) 数组，包含[x, y, heading]
        format_type: 输出格式 ('list', 'string', 'detailed')
    """
    if waypoints.shape[0] != 6 or waypoints.shape[1] != 3:
        raise ValueError(f"Expected shape (6, 3), got {waypoints.shape}")
    
    if format_type == 'list':
        # 返回Python列表
        return waypoints.tolist()
    
    elif format_type == 'string':
        # 返回格式化字符串
        lines = []
        for i, (x, y, h) in enumerate(waypoints):
            time = i * 0.5
            lines.append(f"t={time:.1f}s: x={x:.6f}, y={y:.6f}, heading={h:.6f}")
        return "\n".join(lines)
    
    elif format_type == 'detailed':
        # 详细描述格式
        lines = ["Trajectory waypoints (relative to current position):"]
        for i, (x, y, h) in enumerate(waypoints):
            time = i * 0.5
            dist = np.sqrt(x**2 + y**2)
            lines.append(f"  Waypoint {i+1} (t={time:.1f}s):")
            lines.append(f"    - x: {x:.6f} meters")
            lines.append(f"    - y: {y:.6f} meters")
            lines.append(f"    - heading: {h:.6f} radians")
            lines.append(f"    - distance: {dist:.6f} meters")
        return "\n".join(lines)
    
    elif format_type == 'compact':
        # 紧凑格式
        points = []
        for i, (x, y, h) in enumerate(waypoints):
            points.append(f"[{x:.6f}, {y:.6f}, {h:.6f}]")
        return "; ".join(points)
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

def convert_nuscenes_to_msswift(args):
    """
    将nuScenes轨迹数据转换为MS-Swift格式
    """
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载pkl文件
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
    
    # 获取数据列表
    data_list = data.get('data_list', [])
    if not data_list:
        raise ValueError("No data found in the input file")
    
    print(f"Found {len(data_list)} samples")
    
    # 如果需要，只转换部分数据
    if args.convert_ratio < 1.0:
        num_samples = int(len(data_list) * args.convert_ratio)
        indices = random.sample(range(len(data_list)), num_samples)
        data_list = [data_list[i] for i in indices]
    else:
        num_samples = len(data_list)
    
    # 生成输出文件名
    if args.output_name:
        output_file = output_dir / args.output_name
    else:
        # 从输入文件名中提取基本名称
        base_name = Path(args.input_file).stem
        if 'train' in base_name or 'train' in args.data_split:
            split = 'train'
        elif 'val' in base_name or 'val' in args.data_split:
            split = 'val'
        else:
            split = args.data_split
        
        output_file = output_dir / f"nuscenes_trajectory_{split}.jsonl"
    
    # 获取系统提示
    system_prompt = get_system_prompt(args.system_prompt_type)
    
    # 相机顺序和描述
    if args.camera_order:
        camera_order = [cam.strip() for cam in args.camera_order.split(',')]
    else:
        # nuScenes默认相机顺序
        camera_order = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
    
    # 相机描述映射
    camera_descriptions = {
        'CAM_FRONT': 'front view',
        'CAM_FRONT_LEFT': 'front-left view',
        'CAM_FRONT_RIGHT': 'front-right view',
        'CAM_BACK': 'rear view',
        'CAM_BACK_LEFT': 'rear-left view',
        'CAM_BACK_RIGHT': 'rear-right view',
        'CAM_SIDE_LEFT': 'left side view',
        'CAM_SIDE_RIGHT': 'right side view'
    }
    
    # 自定义描述
    if args.custom_descriptions:
        for cam_desc in args.custom_descriptions.split(','):
            if ':' in cam_desc:
                cam, desc = cam_desc.split(':', 1)
                camera_descriptions[cam] = desc.strip()
    
    converted_samples = []
    error_count = 0
    no_trajectory_count = 0
    no_images_count = 0
    
    for i, sample in enumerate(tqdm(data_list, desc=f"Converting {args.data_split} data")):
        try:
            # 检查是否有轨迹数据
            if 'gt_planning' not in sample or sample['gt_planning'] is None:
                no_trajectory_count += 1
                if args.verbose and no_trajectory_count <= 5:
                    print(f"Warning: No trajectory data for sample {sample.get('token', f'index_{i}')}")
                continue
            
            # 获取轨迹数据
            trajectory = sample['gt_planning']
            if trajectory.shape[0] > 1:  # 如果有多个轨迹，取第一个
                trajectory = trajectory[0]
            
            # 检查轨迹形状
            if trajectory.shape != (6, 3):
                if args.verbose:
                    print(f"Warning: Unexpected trajectory shape {trajectory.shape} for sample {sample.get('token', f'index_{i}')}")
                # 尝试reshape
                if trajectory.size == 18:  # 总共18个元素
                    trajectory = trajectory.reshape(6, 3)
                else:
                    error_count += 1
                    continue
            
            # 获取图片信息
            image_data = sample.get('images', {})
            
            # 构建图片路径列表（按指定顺序）
            image_paths = []
            available_cameras = []
            
            for camera in camera_order:
                if camera in image_data and image_data[camera]:
                    # 获取图片路径
                    img_rel_path = image_data[camera]
                    
                    # 构建完整路径
                    if args.img_root:
                        # 检查路径是否存在
                        possible_paths = []
                        if isinstance(img_rel_path, list):
                            for path in img_rel_path:
                                full_path = str(Path(args.img_root) / path)
                                if os.path.exists(full_path):
                                    possible_paths.append(full_path)
                                    break
                        else:
                            possible_paths.append(str(Path(args.img_root) / img_rel_path))
                        
                        if possible_paths:
                            img_full_path = possible_paths[0]
                        else:
                            # 尝试常见路径模式
                            base_name = Path(str(img_rel_path)).name
                            search_patterns = [
                                f"**/{base_name}",
                                f"samples/{camera}/{base_name}",
                                f"{camera}/{base_name}"
                            ]
                            found = False
                            for pattern in search_patterns:
                                matches = list(Path(args.img_root).rglob(pattern))
                                if matches:
                                    img_full_path = str(matches[0])
                                    found = True
                                    break
                            if not found:
                                if args.verbose and no_images_count < 5:
                                    print(f"Warning: Image not found for {camera}: {img_rel_path}")
                                continue
                    else:
                        img_full_path = img_rel_path
                    
                    # 检查图片是否存在
                    if args.check_images and img_full_path.startswith('/'):
                        if not os.path.exists(img_full_path):
                            if args.verbose and no_images_count < 5:
                                print(f"Warning: Image file not found: {img_full_path}")
                                no_images_count += 1
                            continue
                    
                    image_paths.append(img_full_path)
                    available_cameras.append(camera)
            
            # 如果没有图片
            if not image_paths and not args.allow_no_images:
                if args.verbose and no_images_count < 5:
                    print(f"Warning: No images found for sample {sample.get('token', f'index_{i}')}")
                    no_images_count += 1
                continue
            
            # 准备元数据
            metadata = {
                'sample_idx': sample.get('sample_idx', i),
                'token': sample.get('token', f'sample_{i}'),
                'timestamp': sample.get('timestamp', 0),
                'scene_name': sample.get('scene_name', ''),
                'location': sample.get('location', ''),
                'description': sample.get('description', ''),
                'has_trajectory': True,
                'num_images': len(image_paths),
                'available_cameras': available_cameras,
                'source_file': Path(args.input_file).name
            }
            
            # 获取场景描述
            scene_description = sample.get('description', '')
            
            # 构建用户提示
            if args.use_scene_description and scene_description:
                user_prompt = f"Scene description: {scene_description}. Based on the current driving scene, please plan a 3-second trajectory with 6 waypoints (x, y, heading) at 0.5-second intervals."
            else:
                user_prompt = "Based on the current driving scene, please plan a 3-second trajectory with 6 waypoints (x, y, heading) at 0.5-second intervals."
            
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            
            # 用户消息
            if image_paths:
                # 构建图像标签
                image_tags = " ".join([f"<image>" for _ in range(len(image_paths))])
                
                # 获取相机描述
                available_descriptions = [camera_descriptions.get(cam, cam.lower().replace('_', ' ') + ' view') 
                                        for cam in available_cameras]
                
                if len(available_descriptions) > 1:
                    last_desc = available_descriptions.pop()
                    camera_desc_str = ", ".join(available_descriptions) + f", and {last_desc}"
                else:
                    camera_desc_str = available_descriptions[0] if available_descriptions else "vehicle's cameras"
                
                # 组合用户消息
                if args.image_first_question:
                    user_message = f"Here are images from the {camera_desc_str}: {image_tags}. {user_prompt}"
                else:
                    user_message = f"{user_prompt} The images are from the {camera_desc_str}: {image_tags}."
            else:
                user_message = user_prompt
            
            messages.append({"role": "user", "content": user_message})
            
            # 助手消息：格式化轨迹
            assistant_message = format_trajectory(trajectory, args.trajectory_format)
            messages.append({"role": "assistant", "content": assistant_message})
            
            # 构建转换后的样本
            converted_sample = {
                "messages": messages,
                "metadata": metadata
            }
            
            # 如果有图片，添加images字段
            if image_paths:
                converted_sample["images"] = image_paths
            
            converted_samples.append(converted_sample)
            
        except Exception as e:
            error_count += 1
            if args.verbose and error_count <= 5:
                print(f"Error processing sample {i}: {e}")
            continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nConversion Summary:")
    print(f"  Total samples: {num_samples}")
    print(f"  Successfully converted: {len(converted_samples)}")
    print(f"  Samples without trajectory: {no_trajectory_count}")
    print(f"  Samples without images: {no_images_count}")
    print(f"  Errors: {error_count}")
    print(f"  Saved to: {output_file}")
    
    # 保存统计信息
    stats_file = output_dir / f"conversion_stats_{args.data_split}.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Trajectory Data Conversion Statistics\n")
        f.write(f"===================================\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Successfully converted: {len(converted_samples)}\n")
        f.write(f"Samples without trajectory: {no_trajectory_count}\n")
        f.write(f"Samples without images: {no_images_count}\n")
        f.write(f"Conversion errors: {error_count}\n")
        f.write(f"Conversion ratio: {args.convert_ratio}\n")
        f.write(f"System prompt type: {args.system_prompt_type}\n")
        f.write(f"Trajectory format: {args.trajectory_format}\n")
        f.write(f"Use scene description: {args.use_scene_description}\n")
        
        if converted_samples:
            # 添加一些示例轨迹统计
            sample_traj = np.array(converted_samples[0]['messages'][-1]['content'].split('\n')[0].split(': ')[1].split(', '))
            f.write(f"\nExample trajectory (first sample):\n")
            f.write(f"  Format: {converted_samples[0]['messages'][-1]['content'][:100]}...\n")
            f.write(f"  Number of images: {converted_samples[0]['metadata']['num_images']}\n")
            f.write(f"  Available cameras: {converted_samples[0]['metadata']['available_cameras']}\n")
    
    return converted_samples

def main():
    parser = argparse.ArgumentParser(description='Convert nuScenes trajectory data to MS-Swift format')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input pkl file with trajectory data')
    parser.add_argument('--output_dir', type=str, 
                       default='./preprocessed_data/nuscenes_trajectory',
                       help='Output directory for converted data')
    parser.add_argument('--data_split', type=str, choices=['train', 'val', 'test'], default='train',
                       help='Data split: train, val, or test')
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                       help='Ratio of data to convert (0.0 to 1.0)')
    parser.add_argument('--img_root', type=str, default='',
                       help='Root directory for images (if images are stored separately)')
    parser.add_argument('--output_name', type=str, default='',
                       help='Output file name (default: auto-generated)')
    parser.add_argument('--system_prompt_type', type=str, default='default',
                       choices=['default', 'detailed', 'planning_focused'],
                       help='Type of system prompt to use')
    parser.add_argument('--camera_order', type=str, default='',
                       help='Comma-separated list of camera order')
    parser.add_argument('--custom_descriptions', type=str, default='',
                       help='Custom camera descriptions, format: CAM1:desc1,CAM2:desc2')
    parser.add_argument('--image_first_question', action='store_true',
                       help='Place image tags at the beginning of the question')
    parser.add_argument('--check_images', action='store_true',
                       help='Check if image files exist')
    parser.add_argument('--allow_no_images', action='store_true',
                       help='Allow samples with no images')
    parser.add_argument('--use_scene_description', action='store_true',
                       help='Include scene description in the prompt')
    parser.add_argument('--trajectory_format', type=str, default='string',
                       choices=['list', 'string', 'detailed', 'compact'],
                       help='Format for trajectory output')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 运行转换
    convert_nuscenes_to_msswift(args)

if __name__ == '__main__':
    main()

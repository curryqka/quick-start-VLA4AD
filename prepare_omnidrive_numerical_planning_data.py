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
        'default': """You are an autonomous driving navigation planner. Your task is to analyze the driving scene from the vehicle's cameras and provide high-level semantic navigation instructions and decision explanations. Based on the current environment, you need to output the overall navigation strategy and explain your reasoning.""",
        
        'detailed': """You are an expert autonomous driving navigation planner. Your responsibilities are:

1. **Perception & Understanding**: Analyze the complete driving scene from all available camera perspectives
2. **Semantic Planning**: Generate high-level navigation instructions based on the current environment
3. **Decision Explanation**: Explain the reasoning behind your navigation decisions considering:
   - Traffic rules and regulations
   - Safety considerations
   - Road geometry and conditions
   - Surrounding traffic participants
   - Short-term and long-term objectives
   
Please provide clear, structured navigation instructions and explain your decision-making process thoroughly.""",
        
        'instruction_focused': """You are a driving strategy assistant. For the given driving scenario, provide:
1. High-level semantic navigation instructions (where to go, which lane to take, what maneuvers to perform)
2. Detailed explanation of your decision-making process
3. Consideration of safety, efficiency, and comfort

Focus on the overall strategy, not detailed trajectory points."""
    }
    
    return prompts.get(prompt_type, prompts['default'])

def get_user_prompt(prompt_type='default'):
    """
    获取用户提示模板
    """
    prompts = {
        'default': "Based on the current driving scene, please provide high-level semantic navigation instructions and explain your decision-making process. What should be the overall driving strategy?",
        
        'structured': """Please analyze the current driving scene and provide:

1. **High-level Navigation Instructions**:
   - Primary objective and destination direction
   - Lane selection and positioning strategy
   - Key maneuvers to perform

2. **Decision Explanation**:
   - Why this strategy is appropriate for the current scene
   - What factors influenced your decision
   - Safety considerations and contingency plans""",
        
        'simple': "Looking at the current driving environment, what navigation strategy would you recommend? Please explain your reasoning."
    }
    
    return prompts.get(prompt_type, prompts['default'])

def format_navigation_instructions(trajectory, description, format_type='structured'):
    """
    基于轨迹和场景描述生成高层级导航指令
    """
    if trajectory.shape[0] != 6 or trajectory.shape[1] != 3:
        raise ValueError(f"Expected trajectory shape (6, 3), got {trajectory.shape}")
    
    # 从轨迹中提取基本信息
    start_x, start_y, start_heading = trajectory[0]
    end_x, end_y, end_heading = trajectory[-1]
    
    # 计算基本移动
    displacement = np.sqrt(end_x**2 + end_y**2)
    heading_change = end_heading - start_heading
    
    # 确定基本动作
    if abs(end_y) < 0.5:  # 主要是直行
        primary_action = "continue straight ahead"
    elif end_y > 0.5:  # 向左
        if heading_change > 0.1:
            primary_action = "turn left"
        else:
            primary_action = "move to the left"
    elif end_y < -0.5:  # 向右
        if heading_change < -0.1:
            primary_action = "turn right"
        else:
            primary_action = "move to the right"
    else:
        primary_action = "proceed forward"
    
    # 计算速度变化
    distances = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i, 0] - trajectory[i-1, 0]
        dy = trajectory[i, 1] - trajectory[i-1, 1]
        distances.append(np.sqrt(dx**2 + dy**2))
    
    avg_speed = np.mean(distances) / 0.5  # 0.5秒间隔
    speed_variation = np.std(distances) / 0.5
    
    # 基于速度描述
    if avg_speed < 2.0:
        speed_description = "maintain a low speed"
    elif avg_speed < 5.0:
        speed_description = "maintain a moderate speed"
    else:
        speed_description = "maintain a relatively high speed"
    
    if speed_variation > 1.0:
        speed_description += " with some acceleration/deceleration"
    else:
        speed_description += " steadily"
    
    # 解析场景描述
    scene_keywords = []
    if description:
        # 简单的关键词提取
        keywords = ['parking', 'lot', 'barrier', 'exit', 'intersection', 'highway', 
                   'urban', 'residential', 'pedestrian', 'crosswalk', 'traffic', 'light',
                   'construction', 'tunnel', 'bridge', 'curve', 'straight']
        scene_words = description.lower().replace(',', ' ').split()
        scene_keywords = [word for word in scene_words if word in keywords]
    
    # 生成导航指令
    if format_type == 'structured':
        instructions = f"""**High-Level Navigation Instructions:**

1. **Primary Action**: {primary_action}
2. **Speed Strategy**: {speed_description}
3. **Trajectory Overview**: Plan to cover approximately {displacement:.1f} meters over 3 seconds
4. **Heading Adjustment**: Adjust heading by {heading_change:.2f} radians

**Decision Explanation:**
"""
        
        if scene_keywords:
            instructions += f"- The scene contains: {', '.join(scene_keywords)}\n"
        
        instructions += f"- Choose to {primary_action} based on the available space and trajectory requirements\n"
        instructions += f"- The speed strategy considers vehicle dynamics and environmental factors\n"
        instructions += f"- The planned displacement of {displacement:.1f}m allows for safe and comfortable maneuvering\n"
        instructions += "- Always prioritize safety while maintaining reasonable progress toward the objective"
    
    elif format_type == 'detailed':
        instructions = f"""**Scene Analysis:**
Current environment: {description or 'General driving scene'}
Detected elements: {', '.join(scene_keywords) if scene_keywords else 'Standard driving conditions'}

**Navigation Strategy:**
1. **Primary Maneuver**: {primary_action}
2. **Speed Profile**: {speed_description}
3. **Path Characteristics**: Gentle path with {displacement:.1f}m total displacement
4. **Orientation**: Heading will change by {heading_change:.2f} radians

**Reasoning:**
- The selected maneuver ({primary_action}) is appropriate for the observed scene configuration
- Speed selection balances safety with efficiency
- The gradual heading change ensures passenger comfort
- Environmental factors have been considered in the strategy formulation
- The plan allows for adjustments if unexpected situations arise"""
    
    else:  # simple
        instructions = f"Based on the current scene ({description}), I recommend to {primary_action} while {speed_description}. The overall strategy covers {displacement:.1f} meters with a heading adjustment of {heading_change:.2f} radians. This approach balances safety with progress toward the objective."
    
    return instructions

def convert_nuscenes_to_msswift(args):
    """
    将nuScenes轨迹数据转换为MS-Swift格式，专注于高层级导航指令
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
        
        output_file = output_dir / f"nuscenes_navigation_{split}.jsonl"
    
    # 获取系统提示
    system_prompt = get_system_prompt(args.system_prompt_type)
    user_prompt_template = get_user_prompt(args.user_prompt_type)
    
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
            
            # 获取场景描述
            scene_description = sample.get('description', '')
            
            # 生成高层级导航指令
            navigation_instructions = format_navigation_instructions(
                trajectory, 
                scene_description,
                args.navigation_format
            )
            
            # 准备元数据
            metadata = {
                'sample_idx': sample.get('sample_idx', i),
                'token': sample.get('token', f'sample_{i}'),
                'timestamp': sample.get('timestamp', 0),
                'scene_name': sample.get('scene_name', ''),
                'location': sample.get('location', ''),
                'description': scene_description,
                'has_trajectory': True,
                'num_images': len(image_paths),
                'available_cameras': available_cameras,
                'source_file': Path(args.input_file).name,
                'trajectory': trajectory.tolist()  # 将轨迹数据存储在metadata中
            }
            
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
                    camera_desc_str = available_descriptions[0] if available_cameras else "vehicle's cameras"
                
                # 组合用户消息
                if args.image_first_question:
                    user_message = f"Here are images from the {camera_desc_str}: {image_tags}. {user_prompt_template}"
                else:
                    user_message = f"{user_prompt_template} The images are from the {camera_desc_str}: {image_tags}."
            else:
                if scene_description and args.include_scene_in_user:
                    user_message = f"Scene: {scene_description}. {user_prompt_template}"
                else:
                    user_message = user_prompt_template
            
            messages.append({"role": "user", "content": user_message})
            
            # 助手消息：高层级导航指令
            messages.append({"role": "assistant", "content": navigation_instructions})
            
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
    
    # 显示示例
    if converted_samples:
        print(f"\nExample converted sample:")
        print(f"  User message: {converted_samples[0]['messages'][1]['content'][:100]}...")
        print(f"  Assistant message: {converted_samples[0]['messages'][2]['content'][:100]}...")
        print(f"  Metadata keys: {list(converted_samples[0]['metadata'].keys())}")
        print(f"  Trajectory shape in metadata: {len(converted_samples[0]['metadata']['trajectory'])}x{len(converted_samples[0]['metadata']['trajectory'][0])}")
        print(f"  Number of images: {converted_samples[0]['metadata']['num_images']}")
    
    # 保存统计信息
    stats_file = output_dir / f"conversion_stats_{args.data_split}.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Navigation Data Conversion Statistics\n")
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
        f.write(f"User prompt type: {args.user_prompt_type}\n")
        f.write(f"Navigation format: {args.navigation_format}\n")
        f.write(f"Include scene in user prompt: {args.include_scene_in_user}\n")
    
    return converted_samples

def main():
    parser = argparse.ArgumentParser(description='Convert nuScenes trajectory data to MS-Swift format with high-level navigation instructions')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input pkl file with trajectory data')
    parser.add_argument('--output_dir', type=str, 
                       default='./preprocessed_data/nuscenes_navigation',
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
                       choices=['default', 'detailed', 'instruction_focused'],
                       help='Type of system prompt to use')
    parser.add_argument('--user_prompt_type', type=str, default='default',
                       choices=['default', 'structured', 'simple'],
                       help='Type of user prompt to use')
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
    parser.add_argument('--navigation_format', type=str, default='structured',
                       choices=['structured', 'detailed', 'simple'],
                       help='Format for navigation instructions output')
    parser.add_argument('--include_scene_in_user', action='store_true',
                       help='Include scene description in user prompt when no images')
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

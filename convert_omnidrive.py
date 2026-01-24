import os
import json
import argparse
import random
from tqdm import tqdm
from pathlib import Path

def get_system_prompt(prompt_type='default'):
    """
    获取系统提示
    """
    prompts = {
        'default': """You are an autonomous driving assistant. Your task is to analyze the driving scene from the vehicle's six cameras and provide helpful responses to questions about the driving environment, traffic conditions, and safety considerations.""",
        
        'detailed': """You are an expert autonomous driving assistant. Your role is to:
1. Analyze the driving scene from six camera perspectives (front, front-left, front-right, rear, rear-left, rear-right)
2. Provide detailed observations about road conditions, traffic, pedestrians, and other relevant elements
3. Answer questions about driving decisions, safety considerations, and environmental factors
4. Help understand complex driving scenarios

Please provide clear, accurate, and helpful responses based on the visual information from the cameras.""",
        
        'simple': """You are a driving assistant. Analyze the scene from the vehicle's cameras and answer questions about the driving environment."""
    }
    
    return prompts.get(prompt_type, prompts['default'])

def convert_omnidrive_to_msswift(args):
    """
    将Omnidrive数据转换为MS-Swift格式
    """
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取输入文件
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 如果设置了转换比例，只转换部分数据
    if args.convert_ratio < 1.0:
        num_samples = int(len(lines) * args.convert_ratio)
        lines = random.sample(lines, num_samples)
    else:
        num_samples = len(lines)
    
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
        
        output_file = output_dir / f"omnidrive_{split}.jsonl"
    
    # 获取系统提示
    system_prompt = get_system_prompt(args.system_prompt_type)
    
    # 定义相机顺序和描述
    if args.camera_order:
        # 如果提供了自定义相机顺序
        camera_order = [cam.strip() for cam in args.camera_order.split(',')]
    else:
        # 默认相机顺序
        camera_order = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    
    # 相机描述映射
    camera_descriptions = {
        'FRONT': 'front view',
        'FRONT_LEFT': 'front-left view', 
        'FRONT_RIGHT': 'front-right view',
        'BACK': 'rear view',
        'BACK_LEFT': 'rear-left view',
        'BACK_RIGHT': 'rear-right view',
        'SIDE_LEFT': 'left side view',
        'SIDE_RIGHT': 'right side view'
    }
    
    # 如果需要自定义描述
    if args.custom_descriptions:
        for cam_desc in args.custom_descriptions.split(','):
            if ':' in cam_desc:
                cam, desc = cam_desc.split(':', 1)
                camera_descriptions[cam] = desc.strip()
    
    converted_samples = []
    error_count = 0
    
    for line in tqdm(lines, desc=f"Converting {args.data_split} data"):
        try:
            data = json.loads(line.strip())
            
            # 准备元数据
            metadata = {
                'uuid': data.get('uuid', ''),
                'clip_id': data.get('clip_id', ''),
                'timestamp': data.get('timestamp', ''),
                'tag': data.get('tag', []),
                'type': data.get('type', ''),
                'qa_version': data.get('qa_version', ''),
                'conversation_length': len(data.get('conversations', [])),
                'source_file': Path(args.input_file).name
            }
            
            # 获取图片路径
            image_data = data.get('image', {})
            
            # 构建图片列表（按指定顺序）
            image_paths = []
            available_cameras = []
            
            for camera in camera_order:
                if camera in image_data and image_data[camera]:
                    # 取第一个图片路径
                    img_rel_path = image_data[camera][0]
                    # 构建完整路径
                    if args.img_root:
                        img_full_path = str(Path(args.img_root) / img_rel_path)
                    else:
                        img_full_path = img_rel_path
                    
                    # 检查图片是否存在（如果设置了检查选项）
                    if args.check_images and img_full_path.startswith('/'):
                        if not os.path.exists(img_full_path):
                            if args.verbose:
                                print(f"Warning: Image not found: {img_full_path}")
                            continue
                    
                    image_paths.append(img_full_path)
                    available_cameras.append(camera)
            
            # 如果没有图片，跳过这个样本
            if not image_paths and not args.allow_no_images:
                if args.verbose:
                    print(f"Warning: No images found for sample {data.get('uuid', 'unknown')}")
                continue
            
            # 获取对话
            conversations = data.get('conversations', [])
            if not conversations:
                if args.verbose:
                    print(f"Warning: No conversations found for sample {data.get('uuid', 'unknown')}")
                continue
            
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            
            # 处理每条对话
            for i, conv in enumerate(conversations):
                if conv['from'] == 'human':
                    if i == 0 and image_paths:  # 第一条用户消息，如果有图片
                        # 构建图像标签
                        image_tags = " ".join([f"<image>" for _ in range(len(image_paths))])
                        
                        # 获取可用的相机描述
                        available_descriptions = [camera_descriptions.get(cam, cam.lower() + ' view') 
                                                for cam in available_cameras]
                        
                        if len(available_descriptions) > 1:
                            last_desc = available_descriptions.pop()
                            camera_desc_str = ", ".join(available_descriptions) + f", and {last_desc}"
                        else:
                            camera_desc_str = available_descriptions[0] if available_descriptions else "vehicle's cameras"
                        
                        # 构建用户消息
                        if args.image_first_question:
                            user_message = f"Here are images from the {camera_desc_str}: {image_tags}. {conv['value']}"
                        else:
                            user_message = f"{conv['value']} Based on the images from the {camera_desc_str}: {image_tags}."
                    else:
                        # 后续用户消息
                        user_message = conv['value']
                    
                    messages.append({"role": "user", "content": user_message})
                
                elif conv['from'] == 'gpt':
                    messages.append({"role": "assistant", "content": conv['value']})
            
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
            if args.verbose:
                print(f"Error processing line: {e}")
            continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nConversion Summary:")
    print(f"  Input samples: {num_samples}")
    print(f"  Converted samples: {len(converted_samples)}")
    print(f"  Errors: {error_count}")
    print(f"  Saved to: {output_file}")
    
    # 保存统计信息
    stats_file = output_dir / f"conversion_stats_{args.data_split}.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Conversion Statistics\n")
        f.write(f"===================\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Total input samples: {num_samples}\n")
        f.write(f"Successfully converted: {len(converted_samples)}\n")
        f.write(f"Conversion errors: {error_count}\n")
        f.write(f"Conversion ratio: {args.convert_ratio}\n")
        f.write(f"System prompt type: {args.system_prompt_type}\n")
    
    return converted_samples

def main():
    parser = argparse.ArgumentParser(description='Convert OmniDrive data to MS-Swift format')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input raw jsonl file')
    parser.add_argument('--output_dir', type=str, 
                       default='./preprocessed_data/OmniDrive',
                       help='Output directory for converted data')
    parser.add_argument('--data_split', type=str, choices=['train', 'val', 'test'], default='train',
                       help='Data split: train, val, or test')
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                       help='Ratio of data to convert (0.0 to 1.0)')
    parser.add_argument('--img_root', type=str, default='/lustre/MLM_evaluator/data/omnidrive',
                       help='Root directory for images')
    parser.add_argument('--output_name', type=str, default='',
                       help='Output file name (default: auto-generated based on data split)')
    parser.add_argument('--system_prompt_type', type=str, default='default',
                       choices=['default', 'detailed', 'simple'],
                       help='Type of system prompt to use')
    parser.add_argument('--camera_order', type=str, default='',
                       help='Comma-separated list of camera order (e.g., FRONT,FRONT_LEFT,FRONT_RIGHT,BACK,BACK_LEFT,BACK_RIGHT)')
    parser.add_argument('--custom_descriptions', type=str, default='',
                       help='Custom camera descriptions, format: CAM1:desc1,CAM2:desc2')
    parser.add_argument('--image_first_question', action='store_true',
                       help='Place image tags at the beginning of the first question')
    parser.add_argument('--check_images', action='store_true',
                       help='Check if image files exist')
    parser.add_argument('--allow_no_images', action='store_true',
                       help='Allow samples with no images (for debugging)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 运行转换
    convert_omnidrive_to_msswift(args)

if __name__ == '__main__':
    main()

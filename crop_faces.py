import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
import math

class FaceCropNode:
    """
    ComfyUI节点：面部检测和裁剪
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "number_of_faces": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 49,
                    "step": 1,
                    "display": "number"
                }),
                "max_faces_per_image": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_faces",)
    FUNCTION = "crop_faces"
    CATEGORY = "image/faces"
    
    def __init__(self):
        # 初始化MediaPipe面部检测
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    
    def get_aspect_ratio_size(self, aspect_ratio, base_size):
        """根据纵横比计算输出尺寸"""
        ratio_map = {
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "16:9": (16, 9),
            "9:16": (9, 16),
            "3:2": (3, 2),
            "2:3": (2, 3)
        }
        
        w_ratio, h_ratio = ratio_map[aspect_ratio]
        
        # 基于base_size计算实际尺寸
        if w_ratio >= h_ratio:
            width = base_size
            height = int(base_size * h_ratio / w_ratio)
        else:
            height = base_size
            width = int(base_size * w_ratio / h_ratio)
            
        return width, height
    
    def detect_faces(self, image_np):
        """检测图片中的所有面部（输入BGR格式）"""
        # 转换为RGB格式（MediaPipe需要）
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # 获取边界框
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_np.shape
                
                # 转换为像素坐标
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # 计算面部中心点
                center_x = x + width // 2
                center_y = y + height // 2
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'center': (center_x, center_y),
                    'confidence': detection.score[0]
                })
        
        return faces
    
    def detect_faces_rgb(self, rgb_image):
        """检测图片中的所有面部（输入RGB格式）"""
        print(f"[DEBUG] detect_faces_rgb: 输入图像形状 {rgb_image.shape}, 数据类型 {rgb_image.dtype}")
        
        # 直接使用RGB图像进行检测
        results = self.face_detection.process(rgb_image)
        print(f"[DEBUG] MediaPipe检测结果: {results.detections is not None}")
        
        faces = []
        if results.detections:
            print(f"[DEBUG] 检测到 {len(results.detections)} 个面部")
            for i, detection in enumerate(results.detections):
                # 获取边界框
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = rgb_image.shape
                
                print(f"[DEBUG] 面部 {i+1} 相对边界框: xmin={bbox.xmin:.3f}, ymin={bbox.ymin:.3f}, width={bbox.width:.3f}, height={bbox.height:.3f}")
                
                # 转换为像素坐标
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # 确保边界框有效
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                print(f"[DEBUG] 面部 {i+1} 像素坐标: x={x}, y={y}, width={width}, height={height}")
                
                # 计算面部中心点
                center_x = x + width // 2
                center_y = y + height // 2
                
                confidence = detection.score[0] if detection.score else 0.5
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'center': (center_x, center_y),
                    'confidence': confidence
                })
                
                print(f"[DEBUG] 面部 {i+1} 置信度: {confidence:.3f}")
        else:
            print("[DEBUG] 没有检测到面部")
        
        return faces
    
    def sort_faces_by_distance_to_center(self, faces, image_shape):
        """按照距离图片中心点的距离排序面部"""
        img_center_x = image_shape[1] // 2
        img_center_y = image_shape[0] // 2
        
        def distance_to_center(face):
            face_center_x, face_center_y = face['center']
            return math.sqrt((face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2)
        
        return sorted(faces, key=distance_to_center)
    
    def sort_faces_left_to_right(self, faces):
        """按照从左到右的顺序排序面部"""
        return sorted(faces, key=lambda face: face['center'][0])
    
    def crop_face(self, image_np, face_info, scale_factor, shift_factor, aspect_ratio):
        """裁剪单个面部"""
        x, y, width, height = face_info['bbox']
        h, w = image_np.shape[:2]
        
        print(f"[DEBUG] crop_face: 原始bbox=({x}, {y}, {width}, {height}), 图像大小=({w}, {h})")
        
        # 检查原始边界框是否有效
        if width <= 0 or height <= 0:
            print(f"[ERROR] 无效的面部边界框: width={width}, height={height}")
            return None
        
        # 计算扩展后的边界框
        center_x = x + width // 2
        center_y = y + height // 2
        
        print(f"[DEBUG] 面部中心: ({center_x}, {center_y})")
        
        # 应用scale_factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 应用shift_factor（相对于面部大小）
        shift_x = int(width * shift_factor)
        shift_y = int(height * shift_factor)
        
        # 计算新的边界框
        new_x = center_x - new_width // 2 + shift_x
        new_y = center_y - new_height // 2 + shift_y
        
        print(f"[DEBUG] 缩放后边界框: ({new_x}, {new_y}, {new_width}, {new_height})")
        
        # 确保边界框在图片范围内
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        
        # 确保不超出图像边界
        if new_x >= w or new_y >= h:
            print(f"[ERROR] 边界框完全超出图像范围: new_x={new_x}, new_y={new_y}")
            return None
        
        # 调整宽度和高度以适应图像边界
        new_width = min(new_width, w - new_x)
        new_height = min(new_height, h - new_y)
        
        # 确保调整后的尺寸仍然有效
        if new_width <= 0 or new_height <= 0:
            print(f"[ERROR] 调整后的尺寸无效: new_width={new_width}, new_height={new_height}")
            return None
        
        print(f"[DEBUG] 最终裁剪区域: ({new_x}, {new_y}, {new_width}, {new_height})")
        
        # 裁剪图片
        try:
            cropped = image_np[new_y:new_y+new_height, new_x:new_x+new_width]
            print(f"[DEBUG] 裁剪结果形状: {cropped.shape}")
        except Exception as e:
            print(f"[ERROR] 裁剪失败: {e}")
            return None
        
        # 检查裁剪结果
        if cropped.size == 0:
            print(f"[ERROR] 裁剪结果为空")
            return None
        
        # 根据纵横比调整尺寸
        # 计算目标尺寸
        base_size = max(new_width, new_height, 224)  # 确保最小尺寸
        target_width, target_height = self.get_aspect_ratio_size(aspect_ratio, base_size)
        
        print(f"[DEBUG] 目标尺寸: {target_width} x {target_height}")
        
        # 调整大小
        try:
            cropped_resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"[DEBUG] 调整大小后形状: {cropped_resized.shape}")
            return cropped_resized
        except Exception as e:
            print(f"[ERROR] 调整大小失败: {e}")
            return None
    
    def crop_faces(self, image, number_of_faces, scale_factor, shift_factor, start_index, max_faces_per_image, aspect_ratio):
        """主要的面部裁剪函数"""
        print(f"[DEBUG] 开始处理图像，输入类型: {type(image)}")
        
        # 将tensor转换为numpy数组
        if isinstance(image, torch.Tensor):
            print(f"[DEBUG] 输入tensor维度: {image.shape}")
            # ComfyUI的图片格式通常是 [batch, height, width, channels]
            if image.dim() == 4:
                image = image.squeeze(0)  # 移除batch维度
                print(f"[DEBUG] 移除batch维度后: {image.shape}")
            
            # 确保tensor在CPU上并转换为numpy
            image_np = image.detach().cpu().numpy()
            print(f"[DEBUG] 转换为numpy后形状: {image_np.shape}, 数据范围: [{image_np.min():.3f}, {image_np.max():.3f}]")
            
            # 检查数据范围并转换
            if image_np.max() <= 1.0:
                # 数据在[0,1]范围内，转换为[0,255]
                image_np = (image_np * 255).astype(np.uint8)
                print(f"[DEBUG] 从[0,1]范围转换为[0,255]")
            else:
                # 数据可能已经在[0,255]范围内
                image_np = image_np.astype(np.uint8)
                print(f"[DEBUG] 直接转换为uint8")
        else:
            image_np = image
            print(f"[DEBUG] 使用原始numpy数组: {image_np.shape}")
        
        print(f"[DEBUG] 最终图像形状: {image_np.shape}, 数据类型: {image_np.dtype}")
        
        # 检查图像是否有效
        if image_np.size == 0:
            print("[ERROR] 输入图像为空")
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)
        
        # MediaPipe需要RGB格式进行检测
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # ComfyUI通常提供RGB格式，直接使用
            rgb_for_detection = image_np.copy()
            print(f"[DEBUG] 使用RGB格式进行面部检测")
        else:
            print(f"[ERROR] 不支持的图像格式: {image_np.shape}")
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)
        
        # 检测面部
        print(f"[DEBUG] 开始面部检测...")
        faces = self.detect_faces_rgb(rgb_for_detection)
        print(f"[DEBUG] 检测到 {len(faces)} 个面部")
        
        if not faces:
            print("[WARNING] 没有检测到面部")
            # 如果没有检测到面部，返回原图的中心裁剪
            h, w = image_np.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            center_crop = image_np[start_y:start_y+size, start_x:start_x+size]
            
            # 调整到目标尺寸
            target_width, target_height = self.get_aspect_ratio_size(aspect_ratio, 224)
            center_crop_resized = cv2.resize(center_crop, (target_width, target_height))
            
            # 转换为tensor
            face_tensor = torch.from_numpy(center_crop_resized.astype(np.float32) / 255.0)
            return (face_tensor.unsqueeze(0),)
        
        # 限制检测到的面部数量
        faces = faces[:max_faces_per_image]
        print(f"[DEBUG] 限制后面部数量: {len(faces)}")
        
        # 按距离中心点排序（用于选择最近的面部）
        faces_by_distance = self.sort_faces_by_distance_to_center(faces, image_np.shape)
        
        # 选择距离中心最近的faces
        selected_faces = faces_by_distance[:number_of_faces]
        print(f"[DEBUG] 选择的面部数量: {len(selected_faces)}")
        
        # 按从左到右排序选中的面部
        selected_faces = self.sort_faces_left_to_right(selected_faces)
        
        # 应用start_index
        if start_index < len(selected_faces):
            selected_faces = selected_faces[start_index:]
        else:
            selected_faces = []
        
        print(f"[DEBUG] 应用start_index后的面部数量: {len(selected_faces)}")
        
        # 裁剪面部
        cropped_faces = []
        for i, face_info in enumerate(selected_faces):
            print(f"[DEBUG] 裁剪第 {i+1} 个面部，bbox: {face_info['bbox']}")
            cropped_face = self.crop_face(image_np, face_info, scale_factor, shift_factor, aspect_ratio)
            if cropped_face is not None:
                print(f"[DEBUG] 成功裁剪面部 {i+1}，大小: {cropped_face.shape}")
                cropped_faces.append(cropped_face)
            else:
                print(f"[WARNING] 面部 {i+1} 裁剪失败")
        
        if not cropped_faces:
            print("[ERROR] 没有成功裁剪的面部")
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)
        
        # 转换回tensor格式
        batch_tensors = []
        for i, face in enumerate(cropped_faces):
            print(f"[DEBUG] 转换面部 {i+1} 为tensor，形状: {face.shape}")
            # 确保数据范围正确
            if face.max() > 1.0:
                face_normalized = face.astype(np.float32) / 255.0
            else:
                face_normalized = face.astype(np.float32)
            
            face_tensor = torch.from_numpy(face_normalized)
            batch_tensors.append(face_tensor)
        
        # 堆叠成batch
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors)
            print(f"[DEBUG] 最终输出tensor形状: {batch_tensor.shape}")
            return (batch_tensor,)
        else:
            print("[ERROR] 没有有效的tensor")
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)

# 节点映射（ComfyUI需要）
NODE_CLASS_MAPPINGS = {
    "FaceCropNode": FaceCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCropNode": "面部检测裁剪"
}

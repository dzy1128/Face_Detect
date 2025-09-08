import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
import math

class FaceCropNode:
    """
    ComfyUI节点：面部检测和裁剪
    参考 ComfyUI-AutoCropFaces 项目实现
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
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
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
                    "min": 1,
                    "max": 100,
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
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    
    def get_aspect_ratio_dimensions(self, aspect_ratio, base_size):
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
        
        # 计算实际尺寸，保持基础大小
        if w_ratio >= h_ratio:
            width = base_size
            height = int(base_size * h_ratio / w_ratio)
        else:
            height = base_size
            width = int(base_size * w_ratio / h_ratio)
            
        return width, height
    
    def detect_faces(self, image_rgb):
        """检测面部并返回边界框"""
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_rgb.shape
                
                # 转换为像素坐标
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)
                
                # 只保留有效的边界框
                if width > 10 and height > 10:
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0] if detection.score else 0.5,
                        'center_x': x + width // 2,
                        'center_y': y + height // 2
                    })
        
        # 按置信度排序
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        return faces
    
    def crop_single_face(self, image_np, face_info, scale_factor, shift_factor, aspect_ratio):
        """裁剪单个面部"""
        x, y, width, height = face_info['bbox']
        h, w = image_np.shape[:2]
        
        # 计算面部中心
        face_center_x = x + width // 2
        face_center_y = y + height // 2
        
        # 计算裁剪尺寸
        face_size = max(width, height)
        crop_size = int(face_size * scale_factor)
        
        # 应用shift_factor调整垂直位置
        # shift_factor=0: 面部在顶部, 0.5: 居中, 1.0: 面部在底部
        vertical_offset = int(crop_size * (shift_factor - 0.5))
        
        # 计算裁剪区域
        crop_x1 = max(0, face_center_x - crop_size // 2)
        crop_y1 = max(0, face_center_y - crop_size // 2 + vertical_offset)
        crop_x2 = min(w, crop_x1 + crop_size)
        crop_y2 = min(h, crop_y1 + crop_size)
        
        # 确保裁剪区域有效
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        
        # 裁剪图像
        cropped = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if cropped.size == 0:
            return None
        
        # 调整到指定纵横比
        current_size = min(cropped.shape[0], cropped.shape[1])
        target_width, target_height = self.get_aspect_ratio_dimensions(aspect_ratio, max(current_size, 224))
        
        # 确保最小尺寸
        if target_width < 64 or target_height < 64:
            target_width = max(target_width, 64)
            target_height = max(target_height, 64)
        
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    def crop_faces(self, image, number_of_faces, scale_factor, shift_factor, start_index, max_faces_per_image, aspect_ratio):
        """主要的面部裁剪函数"""
        
        # 转换tensor为numpy数组
        if isinstance(image, torch.Tensor):
            # ComfyUI tensor格式通常是 [batch, height, width, channels]
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # 转换为numpy并确保数据范围正确
            if image.is_cuda:
                image = image.cpu()
            
            image_np = image.numpy()
            
            # 检查数据范围
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        else:
            image_np = image
        
        # 确保是3通道图像
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            print(f"[ERROR] 不支持的图像格式: {image_np.shape}")
            # 返回一个黑色图像作为fallback
            fallback = np.zeros((224, 224, 3), dtype=np.uint8)
            tensor_output = torch.from_numpy(fallback.astype(np.float32) / 255.0).unsqueeze(0)
            return (tensor_output,)
        
        # 检测面部
        faces = self.detect_faces(image_np)
        
        if not faces:
            print("[WARNING] 没有检测到面部，返回中心裁剪")
            # 没有检测到面部时返回中心裁剪
            h, w = image_np.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            center_crop = image_np[start_y:start_y+size, start_x:start_x+size]
            
            # 调整到目标尺寸
            target_width, target_height = self.get_aspect_ratio_dimensions(aspect_ratio, 224)
            center_crop_resized = cv2.resize(center_crop, (target_width, target_height))
            
            # 转换为tensor
            tensor_output = torch.from_numpy(center_crop_resized.astype(np.float32) / 255.0).unsqueeze(0)
            return (tensor_output,)
        
        # 限制面部数量
        faces = faces[:max_faces_per_image]
        
        # 应用start_index，使用循环索引
        if start_index >= len(faces):
            start_index = start_index % len(faces)
        
        # 选择要裁剪的面部
        selected_faces = []
        for i in range(number_of_faces):
            face_idx = (start_index + i) % len(faces)
            if face_idx < len(faces):
                selected_faces.append(faces[face_idx])
            else:
                break
        
        # 裁剪面部
        cropped_faces = []
        for face_info in selected_faces:
            cropped_face = self.crop_single_face(image_np, face_info, scale_factor, shift_factor, aspect_ratio)
            if cropped_face is not None:
                cropped_faces.append(cropped_face)
        
        if not cropped_faces:
            print("[ERROR] 面部裁剪失败，返回原图中心裁剪")
            h, w = image_np.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            center_crop = image_np[start_y:start_y+size, start_x:start_x+size]
            
            target_width, target_height = self.get_aspect_ratio_dimensions(aspect_ratio, 224)
            center_crop_resized = cv2.resize(center_crop, (target_width, target_height))
            
            tensor_output = torch.from_numpy(center_crop_resized.astype(np.float32) / 255.0).unsqueeze(0)
            return (tensor_output,)
        
        # 转换为tensor格式
        batch_tensors = []
        for face in cropped_faces:
            # 确保数据在正确范围内
            face_normalized = face.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_normalized)
            batch_tensors.append(face_tensor)
        
        # 堆叠成batch
        if len(batch_tensors) == 1:
            result_tensor = batch_tensors[0].unsqueeze(0)
        else:
            result_tensor = torch.stack(batch_tensors)
        
        return (result_tensor,)

# 节点映射（ComfyUI需要）
NODE_CLASS_MAPPINGS = {
    "FaceCropNode": FaceCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCropNode": "面部检测裁剪"
}
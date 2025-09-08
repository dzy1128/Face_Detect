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
        """检测图片中的所有面部"""
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
        
        # 计算扩展后的边界框
        center_x = x + width // 2
        center_y = y + height // 2
        
        # 应用scale_factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 应用shift_factor（相对于面部大小）
        shift_x = int(width * shift_factor)
        shift_y = int(height * shift_factor)
        
        # 计算新的边界框
        new_x = center_x - new_width // 2 + shift_x
        new_y = center_y - new_height // 2 + shift_y
        
        # 确保边界框在图片范围内
        h, w, _ = image_np.shape
        new_x = max(0, min(new_x, w - 1))
        new_y = max(0, min(new_y, h - 1))
        new_width = min(new_width, w - new_x)
        new_height = min(new_height, h - new_y)
        
        # 裁剪图片
        cropped = image_np[new_y:new_y+new_height, new_x:new_x+new_width]
        
        # 根据纵横比调整尺寸
        if cropped.size > 0:
            # 计算目标尺寸
            base_size = max(new_width, new_height)
            target_width, target_height = self.get_aspect_ratio_size(aspect_ratio, base_size)
            
            # 调整大小
            cropped_resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            return cropped_resized
        
        return None
    
    def crop_faces(self, image, number_of_faces, scale_factor, shift_factor, start_index, max_faces_per_image, aspect_ratio):
        """主要的面部裁剪函数"""
        # 将tensor转换为numpy数组
        if isinstance(image, torch.Tensor):
            # ComfyUI的图片格式通常是 [batch, height, width, channels]
            if image.dim() == 4:
                image = image.squeeze(0)  # 移除batch维度
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image
        
        # 确保是BGR格式（OpenCV格式）
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # 假设输入是RGB，转换为BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 检测面部
        faces = self.detect_faces(image_np)
        
        if not faces:
            # 如果没有检测到面部，返回空的tensor
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)
        
        # 限制检测到的面部数量
        faces = faces[:max_faces_per_image]
        
        # 按距离中心点排序（用于选择最近的面部）
        faces_by_distance = self.sort_faces_by_distance_to_center(faces, image_np.shape)
        
        # 选择距离中心最近的faces
        selected_faces = faces_by_distance[:number_of_faces]
        
        # 按从左到右排序选中的面部
        selected_faces = self.sort_faces_left_to_right(selected_faces)
        
        # 应用start_index
        if start_index < len(selected_faces):
            selected_faces = selected_faces[start_index:]
        else:
            selected_faces = []
        
        # 裁剪面部
        cropped_faces = []
        for face_info in selected_faces:
            cropped_face = self.crop_face(image_np, face_info, scale_factor, shift_factor, aspect_ratio)
            if cropped_face is not None:
                cropped_faces.append(cropped_face)
        
        if not cropped_faces:
            # 如果没有成功裁剪的面部，返回空tensor
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)
        
        # 转换回tensor格式
        batch_tensors = []
        for face in cropped_faces:
            # 转换BGR到RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # 归一化到[0,1]范围
            face_tensor = torch.from_numpy(face_rgb.astype(np.float32) / 255.0)
            batch_tensors.append(face_tensor)
        
        # 堆叠成batch
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors)
            return (batch_tensor,)
        else:
            empty_tensor = torch.zeros((1, 224, 224, 3), dtype=torch.float32)
            return (empty_tensor,)

# 节点映射（ComfyUI需要）
NODE_CLASS_MAPPINGS = {
    "FaceCropNode": FaceCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCropNode": "面部检测裁剪"
}

"""
Face Detection and Cropping Node for ComfyUI
面部检测和裁剪节点
"""

from .crop_faces import FaceCropNode

# 导出节点类映射
NODE_CLASS_MAPPINGS = {
    "FaceCropNode": FaceCropNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCropNode": "面部检测裁剪"
}

# 插件信息
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

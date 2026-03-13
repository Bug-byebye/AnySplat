"""
DifixPipeline 持久化服务模块
在内存中保持模型实例，避免重复加载

使用方式：
    from difix3d_service import infer
    output = infer(input_image, ref_image, prompt, **kwargs)
"""

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageOps

# 尝试导入 DifixPipeline
try:
    from difix3d.src.pipeline_difix import DifixPipeline
except ImportError:
    print("Warning: Could not import DifixPipeline. Make sure difix3d package is available.")


class DifixService:
    """DifixPipeline 单例管理器"""
    
    _instance = None
    _pipe = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化服务（仅第一次调用时真正初始化）"""
        if self._pipe is None:
            self._initialize_model()
    
    def _initialize_model(self):
        """加载模型到内存"""
        print("Loading DifixPipeline model...")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = DifixPipeline.from_pretrained(
            "nvidia/difix_ref", 
            trust_remote_code=True
        )
        self._pipe.to(self._device)
        print(f"Model loaded on {self._device}")
    
    def infer(
        self,
        input_image: Image.Image,
        ref_image: Image.Image,
        prompt: str,
        num_inference_steps: int = 1,
        timesteps: Optional[list] = None,
        guidance_scale: float = 0.0,
    ) -> Image.Image:
        """
        执行推理
        
        Args:
            input_image: 输入图像
            ref_image: 参考图像
            prompt: 提示词
            num_inference_steps: 推理步数
            timesteps: 时间步列表
            guidance_scale: 引导尺度
            
        Returns:
            输出图像
        """
        if self._pipe is None:
            self._initialize_model()
        
        if timesteps is None:
            timesteps = [199]
        
        output = self._pipe(
            prompt,
            image=input_image,
            ref_image=ref_image,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        return output
    
    def to_device(self, device: str):
        """切换模型设备"""
        if self._pipe is not None:
            self._pipe.to(device)
            self._device = device
            print(f"Model moved to {device}")
    
    def clear_cache(self):
        """清理 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局服务实例
_service = DifixService()


def difix_infer(
    input_image: Image.Image,
    ref_image: Image.Image,
    prompt: str,
    num_inference_steps: int = 1,
    timesteps: Optional[list] = None,
    guidance_scale: float = 0.0,
) -> Image.Image:
    """
    便捷推理函数（推荐使用）
    
    使用示例：
        from PIL import Image
        from difix3d_service import infer
        
        input_img = Image.open("input.png")
        ref_img = Image.open("ref.png")
        output = difix_infer(input_img, ref_img, "remove degradation")
        output.save("output.png")
    """
    return _service.infer(
        input_image,
        ref_image,
        prompt,
        num_inference_steps=num_inference_steps,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
    )


def load_image(image_path: str | Path) -> Image.Image:
    """加载并规范化图像"""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def get_service() -> DifixService:
    """获取服务实例（用于高级操作）"""
    return _service

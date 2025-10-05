"""配置管理工具"""
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """配置管理类"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict or {}
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            Config实例
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """字典式访问"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """字典式设置"""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()


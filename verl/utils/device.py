import logging
from typing import Optional, Dict, Type, Protocol, runtime_checkable
from threading import Lock
import torch

logger = logging.getLogger(__name__)

@runtime_checkable
class DeviceStrategy(Protocol):
    """Device strategy protocol"""
    @classmethod
    def is_available(cls) -> bool:
        """Check device availability"""
        ...
    
    @classmethod
    def device_name(cls) -> str:
        """Get device name"""
        ...
    
    @classmethod
    def resource_name(cls) -> str:
        """Get resource name for scheduling"""
        ...
    
    @classmethod
    def get_torch_device(cls) -> any:
        """Get torch device module"""
        ...

class CudaStrategy:
    """NVIDIA GPU strategy"""
    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()
    
    @classmethod
    def device_name(cls) -> str:
        return "cuda"
    
    @classmethod
    def resource_name(cls) -> str:
        return "GPU"
    
    @classmethod
    def get_torch_device(cls) -> any:
        return torch.cuda

class NpuStrategy:
    """Ascend NPU strategy"""
    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch_npu    # noqa: F401
            return torch.npu.is_available()
        except ImportError:
            return False
    
    @classmethod
    def device_name(cls) -> str:
        return "npu"
    
    @classmethod
    def resource_name(cls) -> str:
        return "NPU"
    
    @classmethod
    def get_torch_device(cls) -> any:
        return torch.npu
    
# 全局设备策略注册表
DEVICE_REGISTRY: Dict[str, Type[DeviceStrategy]] = {
    "npu": NpuStrategy,
    "cuda": CudaStrategy,
}

class DeviceManager:
    _instance = None
    _lock = Lock()
    _initialized = False

    @classmethod
    def get_instance(cls) -> 'DeviceManager':
        with cls._lock:
            if not cls._instance:
                cls._instance = cls.__new__(cls)
                cls._instance.__init__()
            return cls._instance

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._strategies = dict(DEVICE_REGISTRY)
        self._device_priority = ["npu", "cuda"]
        self._current_strategy = self._auto_detect()
        
        self._initialized = True
    
    def register_strategy(self, name: str, strategy: Type[DeviceStrategy], priority: int = 0):
        self._strategies[name] = strategy
        self._device_priority.insert(priority, name)
    
    def _auto_detect(self) -> Type[DeviceStrategy]:
        for name in self._device_priority:
            strategy_cls = self._strategies.get(name)
            if strategy_cls is not None:
                if strategy_cls().is_available():
                    logger.info(f"Using device strategy: {name}")
                    return strategy_cls
        raise RuntimeError("No available device found")
    
    @property
    def current_device(self) -> str:
        return self._current_strategy.device_name()
    
    @property
    def resource_name(self) -> str:
        return self._current_strategy.resource_name()
    
    def get_torch_device_module(self):
        return self._current_strategy.get_torch_device()
    
    def get_device(self, device_name: Optional[str] = None) -> torch.device:
        if device_name:
            strategy_cls = self._strategies.get(device_name)
            if strategy_cls is None or not strategy_cls().is_available():
                raise ValueError(f"Device {device_name} not available")
            return torch.device(strategy_cls.device_name())
        return torch.device(self.current_device)


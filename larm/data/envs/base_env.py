from abc import ABC, abstractmethod
from typing import Literal, Dict, Tuple

class BaseEnv(ABC):
    ENV_CARD: Literal["STATIC", "DYNAMIC"] = None

    def __init__(self, config):
        self.config = config
    
    @classmethod
    @abstractmethod
    def _accuracy_reward(cls, **kwargs):
        ...
  
    @classmethod
    @abstractmethod
    def _format_reward(cls, **kwargs):
        ...
    
    @classmethod
    def get_reward_func(cls, func_name: Literal["accuracy", "format"]):
        if func_name == "accuracy":
            return cls._accuracy_reward
       
        elif func_name == "format":
            return cls._format_reward
        
        else:
            raise ValueError(f"Unsupported reward func: {func_name}")

class StaticEnv(BaseEnv):
    ENV_CARD = "STATIC"
    

class DynamicEnv(BaseEnv):
    ENV_CARD = "DYNAMIC"
    
    @abstractmethod
    def set_env(self, task_config: Dict) -> Tuple[str, str]:  
        ...

    @classmethod
    @abstractmethod
    def preprocess_action(self, action: str) -> str:
        ...
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, bool]:
        ...
    
    @abstractmethod
    def feedback(self) -> Tuple[float, bool]:
        ...
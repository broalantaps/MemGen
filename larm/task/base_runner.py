from datasets import DatasetDict
from typing import Dict
from abc import ABC, abstractmethod
from transformers import (
    PreTrainedTokenizerBase
)
import shutil
import os
import glob
from torch.utils.tensorboard import SummaryWriter

from larm.common.config import Config

from .base_model import BaseModel


class BaseRunner(ABC):

    def __init__(
        self,
        model: BaseModel,
        processing_class: PreTrainedTokenizerBase,
        datasets_dict: DatasetDict,  
        configs: Config,   
        env_and_gen_dict: Dict
    ):  
        self.model = model
        self.configs = configs
        
        # parse dataset
        assert len(datasets_dict) == 1
        self.dataset_name = list(datasets_dict.keys())[0]  
        self.dataset_dict: DatasetDict = datasets_dict[self.dataset_name]

        # prepare env
        assert len(env_and_gen_dict) == 1
        env_name = list(env_and_gen_dict.keys())[0]
        self.env_cls, self.gen_cls = env_and_gen_dict[env_name]
        
        # build chat template
        self.processing_class = processing_class


    @abstractmethod
    def train(self):
        raise NotImplementedError("Should be implemented by subclasses")
    
    def evaluate(self):
        evaluate_func_mapping = {
            "STATIC": self._static_evaluate,
            "DYNAMIC": self._dynamic_evaluate
        }
        evaluate_func = evaluate_func_mapping.get(self.env.ENV_CARD)
        if evaluate_func is None:
            raise ValueError("The env has unrecogonized ENV_CARD attribute")
        
        return evaluate_func()


    @abstractmethod
    def _static_evaluate(self):
        raise NotImplementedError("Should be implemented by subclasses")

    
    @abstractmethod
    def _dynamic_evaluate(self):
        raise NotImplementedError("Should be implemented by subclasses")
    
    def _create_tensorboard(self, mode: str):

        log_dir = os.path.join(self.save_dir, "runs")
        writer = SummaryWriter(log_dir=log_dir)
        return writer
    
    def _remove_trainer_ckpts(self, output_dir: str):
        ckpt_paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        for ckpt in ckpt_paths:
            shutil.rmtree(ckpt, ignore_errors=True)
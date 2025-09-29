import os
import sys

from omegaconf import OmegaConf

from larm.common.registry import registry

from larm.data.builders import *
from larm.memory_generator import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))


registry.register_path("library_root", root_dir)  
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)     
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)  
results_root = os.path.join(repo_root, default_cfg.env.results_root)
registry.register_path("results_root", results_root)

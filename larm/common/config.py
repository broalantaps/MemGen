"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from omegaconf import OmegaConf
import json

from larm.common.registry import registry

class Config:
    def __init__(self, args):
        self.config = {}
        
        self.args = args
        
        # Register the config and configuration for setup
        registry.register("configuration", self)

        user_config = self._build_opt_list(self.args.options)   # 命令行指定

        config = OmegaConf.load(self.args.cfg_path)             # 配置文件指定
        runner_config = self.build_runner_config(config, **user_config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config, **user_config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(  # 优先级: 从左向右增加
            runner_config, model_config, dataset_config, user_config
        )
        self.method = config.method
    

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)
    
    @staticmethod
    def build_model_config(config, **kwargs):
        return {"model": config.model}

    @staticmethod
    def build_runner_config(config, **kwargs):
        return {"run": config.run}

    @staticmethod
    def build_dataset_config(config, **kwargs):
        from larm.data.builders.base_builder import BaseDatasetBuilder

        datasets = config.get("datasets", None)
        if datasets is None:
            raise KeyError(
                "Expecting 'datasets' as the root key for dataset configuration."
            )

        dataset_config = OmegaConf.create()

        for dataset_name in datasets:   # 依次处理每一个数据集
            builder_cls: BaseDatasetBuilder = registry.get_builder_class(dataset_name)
            
            # raw dataset configs
            dataset_config_type = datasets[dataset_name].get("type", "default")   # 一般都是 default, 保留此接口可能为了后续调整数据集
            dataset_config_path = builder_cls.default_config_path(
                type=dataset_config_type
            )

            dataset_config = OmegaConf.merge(
                dataset_config,
                OmegaConf.load(dataset_config_path),
                {"datasets": {dataset_name: config["datasets"][dataset_name]}}
            )
        
        return dataset_config
    
    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]
    
    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))
    
    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)  # 转换成 python 容器
        return json.dumps(container, indent=4, sort_keys=True)  # json 格式化字符串

    def to_dict(self):
        return OmegaConf.to_container(self.config)
from larm.common.config import Config
from larm.common.registry import registry


class Task:
    def __init__(self, config: Config, wandb=None):
        self.config = config
        self.wandb = wandb
    
    def build_env_and_generator(self):

        env_and_gms = dict()

        datasets_config = self.config.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config) 
            env_cls = builder.get_env_cls()
            generation_manager_cls = builder.get_generation_manager_cls()

            env_and_gms[name] = (env_cls, generation_manager_cls)

        return env_and_gms        

    def build_dataset(self):
        datasets = dict()
        
        datasets_config = self.config.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config) 
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets
    

    def build_model(self):
        model_config = self.config.model_cfg

        model_cls = registry.get_model_class(self.config.method)

        model = model_cls.from_config(model_config)

        return model


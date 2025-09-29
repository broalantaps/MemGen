"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

class Registry:
    mapping = {
        "model_name_mapping": {},
        "runner_name_mapping": {},
        "builder_name_mapping": {},
        "env_name_mapping": {},
        "state": {},
        "paths": {}
    }
    
    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from larm.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def register_model(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from larm.common.registry import registry
        """

        def wrap(model_cls):
            from larm.task import BaseModel
            assert issubclass(
                model_cls, BaseModel
            ), "All models must inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap
    
    @classmethod
    def register_runner(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from larm.common.registry import registry
        """

        def wrap(runner_cls):
            from larm.task import BaseRunner
            assert issubclass(
                runner_cls, BaseRunner
            ), "All runners must inherit BaseTrainer class"
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(                    
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls
        
        return wrap
    
    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from lavis.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path
    
    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the builder will be registered.

        Usage:

            from lavis.common.registry import registry
            from lavis.datasets.base_dataset_builder import BaseDatasetBuilder
        """

        def wrap(builder_cls):
            from larm.data.builders.base_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class, found {}".format(
                builder_cls
            )
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap
    
    @classmethod
    def register_env(cls, name):

        def wrap(env_cls):
            from larm.data.envs.base_env import BaseEnv

            assert issubclass(
                env_cls, BaseEnv
            ), "All environments must inherit BaseEnv class, found {}".format(
                env_cls
            )
            if name in cls.mapping["env_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["env_name_mapping"][name]
                    )
                )
            cls.mapping["env_name_mapping"][name] = env_cls
            return env_cls

        return wrap

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)
    
    @classmethod
    def get_env_class(cls, name):
        return cls.mapping["env_name_mapping"].get(name, None)
    
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)
    
    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)
    

registry = Registry()
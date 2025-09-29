import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    Cache
)
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack

from typing import Optional
from abc import ABC, abstractmethod
from peft import PeftConfig, get_peft_model

class Trigger(torch.nn.Module, ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, **kwargs) -> bool:
        ...


class NanoTrigger(torch.nn.Module):
    def __init__(self):
        super().__init__()  
        self.register_buffer("_device", torch.tensor(0.0))
    
    @property
    def device(self):
        return self._device.device
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> bool:
        # This "nano trigger" always predicts insertion.
        # It outputs logits where the probability of insertion (index=1) is set to 1.0
        # for every token position in the batch.
        batch_size, seq_len = input_ids.shape

        logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)
        logits[..., 1] = 1.0   
        return logits


class MemGenTrigger(torch.nn.Module):
    """
    Trigger module for the MemGen Model.
    - Input: the weaver receives `inputs_embeds` from the reasoner model's current decoding sequence.
    - Output: the weaver produces a sequence of hidden states with length K, 
      which are concatenated to the original `inputs_embeds` to alter the reasoner's decoding path.
    """
    def __init__(
        self, 
        pretrained_model_name_or_path: str, 
        peft_config: Optional[PeftConfig] = None
    ):
        super().__init__()
        
        # base model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        self.model = self._postprocess(self.model)
        if peft_config is not None:
            self.model = get_peft_model(self.model, peft_config)
        
        self.config = self.model.config

    @property
    def device(self):
        return self.model.device
    
    def _postprocess(self, model: PreTrainedModel):
        for parameter in model.parameters():
            parameter.requires_grad = True
        
        # Replace lm_head with a binary classification head
        hidden_size = model.config.hidden_size
        classification_head = nn.Linear(hidden_size, 2)   
        model.lm_head = classification_head
        
        # Ensure the new head parameters are trainable
        for param in model.lm_head.parameters():
            param.requires_grad = True

        return model

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Trigger decision mechanism for sequence generation.

        The trigger determines its decision based on the already generated `input_ids`.  
        It is influenced by the data distribution but is independent of the weaver module.

        Args:
            input_ids (Optional[torch.LongTensor]): Token IDs of the generated sequence.
            attention_mask (Optional[torch.Tensor]): Attention mask to avoid attending to padding tokens. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model.

        Returns:
            torch.Tensor: Logits tensor of shape `(batch_size, seq_len, num_classes)`,
                        representing the trigger's decision probabilities.
        """   
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs
        ).logits
import torch
from typing import Dict, List, Union, Optional

from dataclasses import dataclass
from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

### Refactor 
@dataclass
class DataCollatorWithPaddingForSeqCls:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    initial_history: str
    # padding: Union[bool, str, PaddingStrategy] = True
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        batch = {k:torch.unsqueeze(v, 1) for k,v in batch.items()}

        # TODO: move history initialization to init OR refactor it here: REFER HaRT code!!
        history = torch.load(self.initial_history) #if initial_history else (torch.zeros(config.n_embd))
        # history = self.history.to(torch.float16) if deepspeed else self.history.float() 
        history = history.float() 
        
        # repeat history for each user, i.e., len(batch['input_ids']) or len(features) or batch['input_ids'].shape[0],
        # and for each token, i.e., len(features[0]['input_ids']) + 1 (eos token), or batch['input_ids'].shape[-1],
        batch['history'] = history.repeat(batch['input_ids'].shape[0], batch['input_ids'].shape[-1], 1)
        
        return batch



@dataclass
class DataCollatorWithPaddingForHaRT:
    """
        Data collator that simply collates batches of lists of dict-like objects 
        and adds padding where none.
        Also, sets other model inputs if passed in args.

        """

    def __init__(self, model_args, config, tokenizer: PreTrainedTokenizerBase, deepspeed=False, is_ft=False, is_user_level_ft=False):
        self.is_ft = is_ft
        self.is_user_level_ft = is_user_level_ft
        self.tokenizer = tokenizer
        self.output_block_last_hidden_states = None if is_ft else model_args.output_block_last_hidden_states 
        if model_args.add_history or config.add_history:
            self.history = torch.load(model_args.initial_history) if model_args.initial_history else (torch.zeros(config.n_embd))
            self.history = self.history.to(torch.float16) if deepspeed else self.history.float() 
            if not is_ft:
                self.layer_ins = model_args.layer_ins if model_args.layer_ins else config.layer_ins
                self.extract_layer = model_args.extract_layer if model_args.extract_layer else config.extract_layer
        else:
            self.history = None 
            self.layer_ins = None
            self.extract_layer = None       

    def __call__(self, examples: List[List[Dict[str, List]]]) -> List[Dict[str, torch.Tensor]]:
        # In this function we'll make the assumption that all `examples` in the batch of lists
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # in the whole batch of lists
        if not isinstance(examples[0], list) or \
        (not self.is_user_level_ft and not isinstance(examples[0][0], (dict, BatchEncoding))) or \
        (self.is_user_level_ft and not isinstance(examples[0][2], (dict, BatchEncoding))):
            raise ValueError("You landed on an incorrect collator! This one's AR_HuLM specific.")

        first = examples[0][2] if self.is_user_level_ft else examples[0][0]
        batch = {}

        if self.is_user_level_ft:
            batch['user_ids'] = torch.tensor([
                                            example[0]
                                            for example in examples
                                            ])
            batch['labels'] = torch.tensor([
                                            example[1]
                                            for example in examples
                                            ])
            # we do this to map it to the examples format as received when not user_level_ft,
            # in order to reuse the rest of the following code for data collation
            blocks = [example[2:] for example in examples] 
            examples = blocks 
        
        

        # Handling all possible keys as figured from the first element        
        for k, v in first.items():
            if k not in ("input_ids", "attention_mask", "labels"):
                raise ValueError("You've landed at an incorrect collator! This one's specific to AR_HuLM.")

            if v is not None and not isinstance(v, str):
                pad = self.tokenizer.eos_token_id if k=='input_ids' else 0 if k=='attention_mask' else -100 
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples]) 
                else:
                    # Running through each example (i.e., each user of the batch, each user will have multiple blocks of words) 
                    batch[k] = torch.tensor([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples
                                            ]) 
        
        block_size = len(first['input_ids'])
        batch['history'] = None if self.history is None else self.history.repeat(len(examples), block_size, 1)
        if not self.is_ft:
            batch['layer_ins'] = self.layer_ins
            batch['extract_layer'] = self.extract_layer
            batch['output_block_last_hidden_states'] = self.output_block_last_hidden_states

        return batch 

def user_default_data_collator(examples: List[List[Dict[str, List]]]) -> List[Dict[str, torch.Tensor]]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``labels``: handles a single value (int or float) per object
        - ``user_id``: handles a single value per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. 
    """

        # In this function we'll make the assumption that all `examples` in the batch of lists
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # in the whole batch of lists
    if not isinstance(examples[0], list) or \
        not isinstance(examples[0][2], (dict, BatchEncoding)):
            raise ValueError("You landed on an incorrect collator! This one's AR_HuLM specific.")
            return
    

    first = examples[0][2]
    batch = {}
    
    batch['user_ids'] = torch.tensor([
                                    example[0]
                                    for example in examples
                                    ])
    batch['labels'] = torch.tensor([
                                    example[1]
                                    for example in examples
                                    ])
    # we do this to map it to the examples format as received when not user_level_ft,
    # in order to reuse the rest of the following code for data collation
    blocks = [example[2:] for example in examples] 
    examples = blocks 
    
    
    # Handling all possible keys as figured from the first element        
    for k, v in first.items():
        if k not in ("input_ids", "attention_mask", "labels"):
            raise ValueError("You've landed at an incorrect collator! This one's specific to AR_HuLM.")
            return
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([be[k] for example in examples for be in example]) 
            else:
                # Running through each example (i.e., each user of the batch, each user will have multiple blocks of words) 
                batch[k] = torch.tensor([be[k] for example in examples for be in example])
    
    return batch 
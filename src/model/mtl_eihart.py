import torch
import torch.nn as nn
from torch.nn import MSELoss

from src.model.modeling_hart import HaRTBasePreTrainedModel, HaRTBaseLMHeadModel
from src.modeling_outputs_eihart import EIHaRTOutput
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

""" EIHaRT model pre-trained for the HuLM task and an auxiliary task (Multi-task learning)  """,

class HistoryMLP(nn.Module):
    def __init__(self, n_state, config):  # in history MLP: n_state=200
        super().__init__()
        nx = config.n_embd
        self.config = config
        self.l_hist = nn.Linear(nx, nx)
        self.l_hs = nn.Linear(nx, nx)
        self.act = ACT2FN["tanh"]

        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, history, hidden_state, sequence_mask):
        h1 = self.l_hist(history)
        h2 = self.l_hs(hidden_state)

        # Fixing the bug where sequence length is -1 when all tokens are padded (i.e. attn_mask is all zeros)
        h2 = h2 * sequence_mask
        # expand along block_len dimension (1) to allow addition with history
        h2 = h2.unsqueeze(1) # [batch_size, 1, embed_dim]

        return self.act(h1 + h2) # [batch_size, block_size, embed_dim]

class MTL_EIHaRTPreTrainedModel(HaRTBasePreTrainedModel):
    def __init__(self, config, hartbaseLMmodel=None):
        super().__init__(config)
        self.config = config
        inner_dim = config.n_inner if config.n_inner is not None else 200
        if hartbaseLMmodel:
            self.transformer = hartbaseLMmodel
        else:
            self.transformer = HaRTBaseLMHeadModel(config)
        if config.add_history:
            self.history_mlp = HistoryMLP(inner_dim, config)

        self.attr_cls_head = AttributeClassificationHead(config)
        
        ## homoscedastic variance param
        self.scale = nn.Parameter(torch.FloatTensor(2)) # one for each task
        self.scale.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        ## weighting losses param
        # self.scale = nn.Linear(2, 2, bias=False) # one for each task
        # self.scale.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
      
    def get_last_pred_token_hidden_state(self, hs, attn_mask):
        batch_size = attn_mask.shape[0]
        
        # finds the last token that is not a padding token in each row.
        sequence_lengths = torch.ne(attn_mask, 0).sum(-1) - 1  # [batch_size]
        
        # selects the indices in sequence_lengths for the respective row in batch, i.e, 
        # finds the embedding of the last non-padded token (indices from sequence_lengths) in each row
        last_pred_token_hs = hs[range(batch_size), sequence_lengths] # [batch_size, embed_dim]
        
        # Fixing the bug where sequence length is -1 when all tokens are padded (i.e. attn_mask is all zeros)
        sequence_mask = (sequence_lengths != -1).int()
        sequence_mask = sequence_mask.unsqueeze(1)
    
        return last_pred_token_hs, sequence_mask

    def save_losses_debugging(self, hulm_loss, ac_loss, mtl_loss):
        import pandas as pd

        
        try:
            if hulm_loss.requires_grad:
                losses = pd.read_csv('/home/nisoni/eihart/EIHaRT/output/debug/normScale_train_age10.csv')
            else:
                losses = pd.read_csv('/home/nisoni/eihart/EIHaRT/output/debug/normScale_eval_age10.csv')
        except:
            losses = pd.DataFrame(columns=['hulm', 'ac', 'mtl'])
        
        losses.loc[-1] = [float(hulm_loss), float(ac_loss), float(mtl_loss)]
        losses.index += 1
        losses = losses.sort_index()

        if hulm_loss.requires_grad:
            losses.to_csv('/home/nisoni/eihart/EIHaRT/output/debug/normScaleSqrtSq_train_age10.csv', index=False)
        else:
            losses.to_csv('/home/nisoni/eihart/EIHaRT/output/debug/normScaleSqrtSq_eval_age10.csv', index=False)


    def forward(
        self,
        input_ids=None,
        # task_ids=None, ## follow-up change (fuc) #1
        user_ids=None,
        ac_labels=None,
        history=None,
        layer_ins=None,
        extract_layer=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_block_last_hidden_states=None,
        output_block_extract_layer_hs=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        layer_ins = layer_ins if layer_ins else self.config.layer_ins
        extract_layer = extract_layer if extract_layer else self.config.extract_layer

        usr_seq_len, blocks_len, block_size = input_ids.shape
        batch_size = usr_seq_len
        batch_loss = torch.tensor(0.0).to(self.device)
        batch_len = 0
        all_blocks_last_hs = () if output_block_last_hidden_states else None
        all_blocks_history = ()
        all_blocks_attn_mask = ()
        all_blocks_extract_layer_hs = () if output_block_extract_layer_hs else None

        for i in range(blocks_len):
            block_input_ids = input_ids[:,i,:]
            block_attention_mask = attention_mask[:,i,:]
            block_labels = labels[:,i,:] if labels is not None else None

            arhulm_output = self.transformer(
                    input_ids=block_input_ids,
                    history=history,
                    layer_ins=layer_ins,
                    extract_layer=extract_layer,
                    past_key_values=past_key_values,
                    attention_mask=block_attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    labels=block_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
            last_block_last_hs = arhulm_output.last_hidden_state
            
            if output_block_last_hidden_states:
                all_blocks_last_hs = all_blocks_last_hs + (arhulm_output.last_hidden_state,)
            
            extract_layer_hs = arhulm_output["extract_layer_hidden_states"][0] if isinstance(arhulm_output, dict) else arhulm_output[-1][0] 
            
            if output_block_extract_layer_hs:
                all_blocks_extract_layer_hs = all_blocks_extract_layer_hs + (extract_layer_hs, )

            if history is not None:
                hs, sequence_mask = self.get_last_pred_token_hidden_state(extract_layer_hs, block_attention_mask)
                history = self.history_mlp(history, hs, sequence_mask)
                all_blocks_history = all_blocks_history + (history[:, 0, :],)
                all_blocks_attn_mask = all_blocks_attn_mask + (sequence_mask, )
                
            if labels is not None:
                batch_loss += arhulm_output["loss"] if isinstance(arhulm_output, dict) else arhulm_output[0] 
                batch_len += len(block_labels[block_labels!= -100])       

        hulm_loss = batch_loss/batch_len if labels is not None else None

        last_updated_history = history
        history_output = (all_blocks_history, all_blocks_attn_mask)

        ac_loss, logits = self.attr_cls_head(input_ids, user_ids, labels=ac_labels, history=history_output, inputs_embeds=inputs_embeds)

        hulm_ppl_loss = hulm_loss 
        # self.save_losses_debugging(hulm_loss, ac_loss, 0)

        # exp/log trick
        # hulm_loss = torch.log(hulm_loss)
        # ac_loss =  torch.log(ac_loss)
        # mtl_loss = torch.exp(hulm_loss + ac_loss) 

        # scaling
        # hulm_loss = self.scale[0] * hulm_loss
        # ac_loss =  self.scale[1] * ac_loss
        # mtl_loss = hulm_loss + ac_loss

        # scaling (squared for positive scale)
        # hulm_loss = self.scale[0] * self.scale[0] * hulm_loss
        # ac_loss =  self.scale[1] * self.scale[1] * ac_loss
        # mtl_loss = (hulm_loss + ac_loss) ** 1/2

        # scaling (sqrt of squared for positive scale and bringing down to regular scale)
        # hulm_loss = ((self.scale[0] * self.scale[0]) ** 1/2) * hulm_loss
        # ac_loss =  ((self.scale[1] * self.scale[1]) ** 1/2) * ac_loss
        # mtl_loss = hulm_loss + ac_loss

        # # homoscedastic loss
        # hulm_log_var = torch.log(self.scale[0] * self.scale[0])
        # ac_log_var = torch.log(self.scale[1] * self.scale[1])
        # hulm_loss = ((torch.exp(-hulm_log_var) * hulm_loss) + hulm_log_var)/2
        # ac_loss =  ((torch.exp(-ac_log_var) * ac_loss) + ac_log_var)/2
        # mtl_loss = hulm_loss + ac_loss

        if labels is not None:
            # homoscedastic loss (for CE and MSE)
            hulm_log_var = torch.log(self.scale[0] * self.scale[0])
            ac_log_var = torch.log(self.scale[1] * self.scale[1])
            hulm_loss = ((torch.exp(-hulm_log_var) * hulm_loss) + hulm_log_var)
            ac_loss =  ((torch.exp(-ac_log_var) * ac_loss) + ac_log_var)/2
            mtl_loss = hulm_loss + ac_loss

        # # homoscedastic loss (scale parameter as variance)
        # hulm_log_var = torch.log(self.scale[0])
        # ac_log_var = torch.log(self.scale[1])
        # hulm_loss = ((torch.exp(-hulm_log_var) * hulm_loss) + hulm_log_var)/2
        # ac_loss =  ((torch.exp(-ac_log_var) * ac_loss) + ac_log_var)/2
        # mtl_loss = hulm_loss + ac_loss

        # linear transformation (equivalent to scaling; so redundant and ignored)
        # hulm_loss, ac_loss = self.scale([hulm_loss, ac_loss])
        # mtl_loss = hulm_loss + ac_loss

        # # simple summing up losses
        # mtl_loss = hulm_loss + ac_loss 
        
        # TODO: only for debugging mode
        # mtl_loss = hulm_loss
        
        if not return_dict:
            output = (last_block_last_hs, last_block_last_hs,) + arhulm_output[3:]
            return ((hulm_loss,) + output) if hulm_loss is not None else output

        ## TODO: return ac_logits, mtl_loss only?
        ## compute_metrics doesn't set prediction_loss_only to true --> that doesn't set
        ## logits to None --> that causes an issue in concating logits.
        return EIHaRTOutput(
            loss=mtl_loss if labels is not None else None, #TODO: tentative labels condition added -- refactor, i guess?!,
            logits=logits,
            hulm_loss=hulm_ppl_loss.repeat(batch_size) if labels is not None else None, #TODO: tentative labels condition added -- refactor, i guess?!
            ac_loss=ac_loss.repeat(batch_size) if ac_labels is not None else None,  #TODO: tentative labels condition added -- refactor, i guess?!
            last_hidden_state=last_block_last_hs,
            all_blocks_last_hidden_states = all_blocks_last_hs,
            all_blocks_extract_layer_hs = all_blocks_extract_layer_hs,
            history=history_output,
            past_key_values=arhulm_output.past_key_values,
            hidden_states=arhulm_output.hidden_states,
            attentions=arhulm_output.attentions,
            cross_attentions=arhulm_output.cross_attentions,
        )

# class MTL_EIHaRTPreTrainedModel(HaRTBasePreTrainedModel):
#     def forward():

#         hulm_inputs, ac_inputs = fetch_inputs_based_on_task_ids()
#         hulm_labels, ac_labels = fetch_labels_based_on_task_ids()
#         for all blocks:
#             hart_outputs = call_hart_lm_head()
#             block_user_states, block_attn_masks, block_last_hidden_states, hart_lm_loss = hart_outputs
#             batch_hart_lm_loss = hart_lm_loss/batch_len
        
#         ac_loss = call_attribute_cls_head(block_user_states, block_attn_masks)

#         eihart_output = (batch_hart_lm_loss, ac_loss, last_hidden_states,... )


class AttributeClassificationHead(nn.Module):
    # _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    ## fix code to generalize initialization
    def __init__(self, config):
        super().__init__()
        # self.freeze_model = config.freeze_model
        self.num_labels = config.ac_num_labels
        self.ac_task = config.ac_task

        ## refeactor both below
        self.use_history_output = True #config.use_history_output
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon) 

        if self.ac_task=='age' or self.ac_task == 'ope':
            self.transform = nn.Linear(config.n_embd, config.n_embd)
        # self.use_hart_no_hist = config.use_hart_no_hist
        # if model_name_or_path:
        #     self.transformer = HaRTPreTrainedModel.from_pretrained(model_name_or_path)
        # elif pt_model:
        #     self.transformer = pt_model
        # else:
        #     self.transformer = HaRTPreTrainedModel(config)
        #     self.init_weights()
        
        # if not self.freeze_model and not self.ac_task=='ope' and not self.ac_task=='user':
        #     self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # if self.ac_task=='age':
        #     self.transform = nn.Linear(config.n_embd, config.n_embd)

        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_pooled_logits(self, logits, input_ids, inputs_embeds):

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                # since we want the index of the last predicted token of the last block only.
                sequence_lengths = sequence_lengths[:, -1]
            else:
                sequence_lengths = -1
                self.logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        # get the logits corresponding to the indices of the last pred tokens (of the last blocks) of each user
        pooled_logits = logits[range(batch_size), sequence_lengths]

        return pooled_logits

    def forward(
        self,
        input_ids=None,
        user_ids=None,
        history=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
          
        if self.ac_task=='user' or self.ac_task=='ope' or self.ac_task=='age':
            if self.use_history_output:
                states = history[0]
                masks = history[1]
                multiplied = tuple(l * r for l, r in zip(states, masks))
                all_blocks_user_states = torch.stack(multiplied, dim=1)
                all_blocks_masks = torch.stack(masks, dim=1)
                sum = torch.sum(all_blocks_user_states, dim=1)
                divisor = torch.sum(all_blocks_masks, dim=1)
                hidden_states = sum/divisor
            else:
                raise ValueError("Since you don't want to use the user-states/history output for a user-level task, please customize the code as per your requirements.")
        else:
            raise ValueError("This code executes for a user-level auxiliary task only.")
       

        ## Refactor logits
        # logits = self.score(hidden_states)  ## or logits = self.score(self.ln_f(hidden_states))

        # logits = self.score(self.ln_f(hidden_states))
        if self.ac_task == 'age' or self.ac_task == 'ope':
            logits = self.score(self.transform(self.ln_f(hidden_states)))
        # if self.finetuning_task=='ope' or self.finetuning_task=='user':
        #     logits = self.score(hidden_states) 
        # elif self.finetuning_task=='age':
        #     self.score(self.transform(self.ln_f(hidden_states)))
        # else:
        #     logits = self.score(self.ln_f(hidden_states))
        
        pooled_logits = logits if (user_ids is None or self.use_history_output) else \
                    self.get_pooled_logits(logits, input_ids, inputs_embeds)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.to(torch.float32).view(-1)) ## fix code for dtype when generalizing
            else:
                raise ValueError("The user-level auxiliary task should perform regression (?).")
                # labels = labels.long()
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        return loss, pooled_logits
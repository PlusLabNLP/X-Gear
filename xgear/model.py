import logging, copy
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Loading pre-trained model {config.model_name}')
        if config.model_name.startswith("google/mt5-"):
            self.model = MT5ForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        elif config.model_name.startswith("copy+google/mt5-"):
            model_name = config.model_name.split('copy+', 1)[1]
            self.model = MT5Copy.from_pretrained(model_name, cache_dir=config.cache_dir, output_attentions=True)
        else:
            raise NotImplementedError
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=1, max_length=100):
        self.eval()
        with torch.no_grad():
            if num_beams == 1:
                self.model._cache_input_ids = batch.enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(batch.enc_idxs.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(batch.enc_idxs.device)
                )
                input_ids = batch.enc_idxs.index_select(0, expanded_return_idx)
                self.model._cache_input_ids = input_ids
                
            outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                          attention_mask=batch.enc_attn, 
                                          num_beams=num_beams, 
                                          max_length=max_length, 
                                          forced_bos_token_id=None)
        
        # decode outputs
        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()

        return final_output

# for constrained decoding
class Prefix_fn_cls():
    def __init__(self, tokenizer, special_tokens, input_enc_idxs):
        self.tokenizer=tokenizer
        self.input_enc_idxs=input_enc_idxs
        self.special_ids = [element for l in self.tokenizer(special_tokens, add_special_tokens=False)['input_ids'] for element in l]
    def get(self, batch_id, previous_token):
        # get input
        inputs = list(set(self.input_enc_idxs[batch_id].tolist()))+self.special_ids
        return inputs
    

# the following is for mT5+copy
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.functional.relu

    elif activation == "gelu":
        return nn.functional.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MyTransformerDecoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        output = tgt

        for mod in self.layers:
            output, wt = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, wt

class MyTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super(MyTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):

        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super(MyTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, wt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)    
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, wt

class MT5Copy(MT5ForConditionalGeneration):
    
    _keys_to_ignore_on_load_missing = ["linear_copy.weight", "linear_copy.bias"]
    
    def __init__(self, config):
        super().__init__(config)
        self.linear_copy = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        
        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == hidden_states.size(0)
        except:
            ipdb.set_trace()
            
        lm_logits = self.lm_head(sequence_output) 
        
        # Copy distribution
        cross_attentions = decoder_outputs['cross_attentions'][-1] # batch x head x decoder_length x encoder_length
        cross_attentions = torch.mean(cross_attentions, dim=1) # batch x decoder_length x encoder_length

        # Probability of copying
        p_copy = torch.sigmoid(self.linear_copy(sequence_output))
        #p_copy = torch.sigmoid(self.linear_copy(torch.cat([sequence_output, attn_output.transpose(1,0)], dim=-1)))
        
        # Merge distribution
        original_word_pro = torch.softmax(lm_logits, dim=-1) * (1 - p_copy) #[batch, sequence_length, vocab_size]
        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, target_length, encoder_length)
        lm_logits = torch.scatter_add(original_word_pro, 2, copy_words, cross_attentions*p_copy)
        
        eps = 1e-7
        lm_logits = torch.log(lm_logits+eps)

        loss = None
        if labels is not None:
            #loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_fct = NLLLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

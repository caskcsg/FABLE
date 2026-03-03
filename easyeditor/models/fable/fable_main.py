import copy
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from ...util import nethook
import torch.optim as optim

import argparse

import numpy as np
import os
from transformers.modeling_attn_mask_utils import AttentionMaskConverter,_prepare_4d_causal_attention_mask
from .fable_hparams import fableHyperParams
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: fableHyperParams,
    layer: int,
    lookup_idxs = None
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(model.device)
    # idxs = [i.sum()-1 for i in input_ids['attention_mask']]
    idxs = [len(i)-1 for i in input_ids['attention_mask']]

    # # wp change
    # if lookup_idxs != None:
    #     idxs = [(mask == 0).sum() + lookup_idx for mask, lookup_idx in zip(input_ids['attention_mask'], lookup_idxs)]
    # else:
    #     idxs = [len(i)-1 for i in input_ids['attention_mask']]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list=[]
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i,idxs[i]])
    zs_out =torch.stack(zs_out_list,dim=0)


    return zs_out,idxs

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # and 'mlp' in n
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters




def apply_fable_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data:list,
    hparams:fableHyperParams,
    copy: bool,
    ex_data:list,
    **kwargs):

    preserve_params = []
    for name, params in model.named_parameters():
        #print(name)
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.sub_layers + hparams.target_layers:
                preserve_params.append(name)
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    z_layer = hparams.sub_layers[-1]
    z_list = []
    lookup_idxs = []
    for index, data in enumerate(batch_data[1:]):
        cur_z, _, lookup_idx = compute_z(   
            model,
            tok,
            data,
            z_layer,
            hparams,
            **kwargs
        )

        z_list.append(cur_z)
        lookup_idxs.extend(lookup_idx)
    
    zs = torch.stack(z_list, dim=0)#(bs,h_dim)


    batch_question = [i['prompt'] for i in batch_data[1:]]
    
    # Insert'
    layer2stat_out = {}
    all_layer = list(set(hparams.sub_layers + hparams.target_layers))
    for layer in all_layer:
        if kwargs['index'] == 241:
            ex_tok = tok(ex_data, padding=True, return_tensors="pt", max_length = 256, truncation = True).to(
                next(model.parameters()).device
            )
        else:
            ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out

        layer2stat_out[layer] = stat_out


    for i, layer in enumerate(hparams.sub_layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks


        cur_zs,idxs = compute_ks(model, tok,batch_question, hparams, z_layer, lookup_idxs)
        
        targets = zs - cur_zs 
        print("z error", torch.linalg.norm(targets, dim=0).mean())


        resid = targets / (len(hparams.sub_layers) - i)  # Distribute residual across layers(1,4096)

        
        if kwargs['index'] == 241:
            ex_tok = tok(ex_data, padding=True, return_tensors="pt", max_length = 256, truncation = True).to(
                next(model.parameters()).device
            )
        else:
            ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input


        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,hparams.lr)
        
        
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        
        for i in range(len(idxs)):
            
            layer_out_ks[i,idxs[i]]+=resid[i]
        
        # from copy import deepcopy
        # first_layer_out_ks = deepcopy(layer_out_ks.detach())
        
        # get_qwen2_causal_mask
        # llama2
        if 'Llama3-8B-Instruct' in hparams.model_name:
            input_causal_mask,input_position_ids,input_cache_position = get_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids,ex_cache_position = get_causal_mask(stat_in,ex_tok['attention_mask'])
        elif 'Qwen2.5-7B-Instruct' in hparams.model_name:
            input_causal_mask,input_position_ids = get_qwen2_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids = get_qwen2_causal_mask(stat_in,ex_tok['attention_mask'])

        for step in range(hparams.optim_num_step):
            #scheduler.step()
            optimizer.zero_grad()

            if 'Qwen2.5-7B-Instruct' in hparams.model_name:
                loss_ex = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids)[0], layer2stat_out[layer])
                
                loss_sub = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids)[0], layer_out_ks)

                loss = loss_ex + loss_sub

            elif 'Llama3-8B-Instruct' in hparams.model_name:
                loss_ex = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], layer2stat_out[layer])
                
                hidden_state_sub = _layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0]
                layer_out_ks_sub = layer_out_ks
                loss_sub = criterion(hidden_state_sub, layer_out_ks_sub)

                loss = loss_ex + loss_sub
    


            print(f'Loss = {loss.item()}, loss_ex = {loss_ex.item()}, loss_sub = {loss_sub.item()}')
            loss.backward(retain_graph=True)
            optimizer.step()

            
            # del loss
            # torch.cuda.empty_cache()

            # print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, hparams.optim_num_step, loss.item(),layer))
            # if loss.item() < 5e-5:
            #     break
        

        for x in [layer_in_ks, layer_out_ks,cur_zs, targets, stat_in]:
            x.cpu()
            del x
        torch.cuda.empty_cache()


    z_layer = hparams.target_layers[-1]
    z_list = []
    for data in [batch_data[0]]:
        cur_z, _, lookup_idx = compute_z(   
            model,
            tok,
            data,
            z_layer,
            hparams,
            **kwargs
        )

        z_list.append(cur_z)
    
    zs = torch.stack(z_list, dim=0)#(bs,h_dim)


    #print(zs.shape)
    batch_question = [i['prompt'] for i in [batch_data[0]]]
    
    sub_question = [d['prompt'] for d in batch_data[1:]]
    sub_answer = [d['target_new'] for d in batch_data[1:]]


    # Insert
    for i, layer in enumerate(hparams.target_layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks



        cur_zs,idxs = compute_ks(model, tok,batch_question, hparams, z_layer)
        
        targets = zs - cur_zs 
        print("z error", torch.linalg.norm(targets, dim=0).mean())


        if kwargs['index'] == 241:
            ex_tok = tok(ex_data, padding=True, return_tensors="pt", max_length = 256, truncation = True).to(
                next(model.parameters()).device
            )
        else:
            ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input


        sub_data = [q + a for q, a in zip(sub_question, sub_answer)]

        sub_tok = tok(sub_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**sub_tok)
                sub_stat_in = tr.input
                sub_stat_out = tr.output
        sub_stat_out = sub_stat_out[0] if type(sub_stat_out) is tuple else sub_stat_out



        resid = targets / (len(hparams.target_layers) - i)  # Distribute residual across layers(1,4096)

        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,hparams.lr)
        
        
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        
        for i in range(len(idxs)):
            
            layer_out_ks[i,idxs[i]]+=resid[i]
        
        # get_qwen2_causal_mask
        # llama2
        if 'Llama3-8B-Instruct' in hparams.model_name:
            input_causal_mask,input_position_ids,input_cache_position = get_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids,ex_cache_position = get_causal_mask(stat_in,ex_tok['attention_mask'])
            sub_causal_mask,sub_position_ids,sub_cache_position = get_causal_mask(sub_stat_in,sub_tok['attention_mask'])
        elif 'Qwen2.5-7B-Instruct' in hparams.model_name:
            input_causal_mask,input_position_ids = get_qwen2_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids = get_qwen2_causal_mask(stat_in,ex_tok['attention_mask'])
            sub_causal_mask,sub_position_ids = get_qwen2_causal_mask(sub_stat_in,sub_tok['attention_mask'])


        for step in range(hparams.optim_num_step):
            #scheduler.step()
            optimizer.zero_grad()

            if 'Qwen2.5-7B-Instruct' in hparams.model_name:
                loss_ex = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids)[0], layer2stat_out[layer])
                
                loss_sub = criterion(_layer(sub_stat_in,attention_mask=sub_causal_mask,position_ids=sub_position_ids)[0], sub_stat_out)
                
                hidden_state_main = _layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids)[0]
                layer_out_ks_main = layer_out_ks
                loss_main = criterion(hidden_state_main, layer_out_ks_main)

                loss = loss_ex + loss_main + loss_sub

            elif 'Llama3-8B-Instruct' in hparams.model_name:
                loss_ex = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], layer2stat_out[layer])
                
                loss_sub = criterion(_layer(sub_stat_in,attention_mask=sub_causal_mask,position_ids=sub_position_ids,cache_position = sub_cache_position)[0], sub_stat_out)
                
                hidden_state_main = _layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0]
                layer_out_ks_main = layer_out_ks
                loss_main = criterion(hidden_state_main, layer_out_ks_main)

                loss = loss_ex + loss_main + loss_sub

            print(f'Loss = {loss.item()}, loss_ex = {loss_ex.item()}, loss_main = {loss_main.item()}, loss_sub = {loss_sub.item()}')
            loss.backward(retain_graph=True)
            optimizer.step()

            # del loss
            # torch.cuda.empty_cache()
        

        for x in [layer_in_ks, layer_out_ks,cur_zs, targets,stat_in]:
            x.cpu()
            del x
        del layer2stat_out
        torch.cuda.empty_cache()

    return model, weights_copy
def get_qwen2_causal_mask(input_tensor,attention_mask,past_key_values_length = 0):
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_tensor.shape[0], input_tensor.shape[1]),
            input_tensor,
            0,
        )

    return attention_mask,position_ids

def get_causal_mask(input_tensor,attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position
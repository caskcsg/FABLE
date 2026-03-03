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
from .unke_hparams import unkeHyperParams
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeHyperParams,
    layer: int,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(model.device)
    # idxs = [i.sum()-1 for i in input_ids['attention_mask']]
    idxs = [len(i)-1 for i in input_ids['attention_mask']]
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




def apply_unke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data:list,
    hparams:unkeHyperParams,
    copy: bool,
    ex_data:list,
    **kwargs):

    preserve_params = []
    for name, params in model.named_parameters():
        #print(name)
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    z_layer = hparams.layers[-1]
    z_list = []
    for data in batch_data:
        
        cur_z, sub_question = compute_z(   
            model,
            tok,
            data,
            z_layer,
            hparams,
            **kwargs
        )

        z_list.append(cur_z)
    
    if hparams.fine == 1:
        zs = cur_z
    else:
        zs = torch.stack(z_list, dim=0)#(bs,h_dim)


    #print(zs.shape)
    batch_question = [i['prompt'] for i in batch_data] + sub_question
    
    # import ipdb; ipdb.set_trace()
    # Insert
    for i, layer in enumerate(hparams.layers):
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
        
        # test_question = [batch_question[0]] + [
        #     "What does George Rankin do for a living?"
        #     "How long has George Rankin been involved in politics?",
        #     "What positions has George Rankin held in politics?",
        #     "What are some political causes that George Rankin has advocated for?",
        #     "What do George Rankin's speeches and interviews primarily focus on?",
        #     "Where is George Rankin frequently quoted?",

        #     # "What type of music does Kathy Saltzman compose?",
        #     # "Which renowned musical groups have performed Kathy Saltzman's work?",
        #     # "Besides composing, what other roles does Kathy Saltzman have in the music industry?",
        #     # "Where has Kathy Saltzman given talks and masterclasses?",
        #     # "How is Kathy Saltzman regarded in the field of composition?"
        # ]
        # test_contexts_tok = tok(test_question, padding=True, return_tensors="pt").to(
        #     next(model.parameters()).device
        # )
        # with torch.no_grad():
        #     with nethook.Trace(
        #         module=model,
        #         layer=hparams.layer_module_tmp.format(hparams.v_loss_layer),
        #         retain_input=True,
        #         retain_output=True,
        #         detach=True,
        #         clone=True,
        #     ) as tr:
        #         _ = model(**test_contexts_tok)
        #         pre = tr.output[0].detach().cpu()


        cur_zs,idxs = compute_ks(model, tok,batch_question, hparams, z_layer)
        
        if hparams.fine == 1:
            targets = zs.repeat(1 + len(sub_question), 1) - cur_zs
        else:
            targets = zs - cur_zs 
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        # torch.save(zs, f'vector/{kwargs["index"]}_zs.pt')
        # torch.save(layer_out_ks, f'vector/{kwargs["index"]}_pre.pt')
        # torch.save(cur_zs, f'vector/{kwargs["index"]}_cur_zs.pt')

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
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out



        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers(1,4096)

        
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
        elif 'Qwen2.5-7B-Instruct' in hparams.model_name:
            # import ipdb; ipdb.set_trace()
            # 注意padding_side
            input_causal_mask,input_position_ids = get_qwen2_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids = get_qwen2_causal_mask(stat_in,ex_tok['attention_mask'])
        
        for step in range(hparams.optim_num_step):
            #scheduler.step()
            optimizer.zero_grad()

            if 'Qwen2.5-7B-Instruct' in hparams.model_name:
                loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids)[0], stat_out) + criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids)[0], layer_out_ks)


                # loss =  criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids)[0], layer_out_ks)
            elif 'Llama3-8B-Instruct' in hparams.model_name:

                loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out) + criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
                # hidden_state = _layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0]
                # attention_mask = contexts_tok['attention_mask']
                # hidden_state_pooled = (hidden_state * attention_mask.unsqueeze(-1)).sum(dim = 1) / attention_mask.sum(dim = 1, keepdim = True)
                # kl_per_sample = 0.5 * (hidden_state_pooled.pow(2).sum(dim = -1))
                # kl_mean = kl_per_sample.mean()

                # loss_mse = criterion(hidden_state, layer_out_ks)
                
                # loss = loss_ex + loss_mse
                # loss = loss_ex + loss_mse + 0 * kl_mean
                # loss = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
                # loss = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0][:,-1], layer_out_ks[:,-1])
                # loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out)+criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0][:,-1], layer_out_ks[:,-1])
        
            # print(f'Loss = {loss.item()}, loss_ex = {loss_ex.item()}, loss_mse = {loss_mse.item()}, kl_mean = {kl_mean.item()}')


            print(f'Loss = {loss.item()}, loss = {loss.item()}')
            loss.backward(retain_graph=True)
            optimizer.step()

            # del loss
            # torch.cuda.empty_cache()

            # print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, hparams.optim_num_step, loss.item(),layer))
            # if loss.item() < 5e-5:
            #     break

        
        # for i in range(layer_in_ks.size(0)):
        #     print(criterion(_layer(layer_in_ks[i].unsqueeze(dim = 0),attention_mask=input_causal_mask[i].unsqueeze(dim = 0),position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks[i].unsqueeze(dim = 0)))

        # with torch.no_grad():
        #     with nethook.Trace(
        #         module=model,
        #         layer=hparams.layer_module_tmp.format(hparams.v_loss_layer),
        #         retain_input=True,
        #         retain_output=True,
        #         detach=True,
        #         clone=True,
        #     ) as tr:
        #         _ = model(**test_contexts_tok)
        #         post = tr.output[0].detach().cpu()
       
       
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from sklearn.manifold import TSNE
        # import math
        # all_data = torch.vstack((pre[:, -1, :], post[:, -1, :])).numpy()
        # y = np.repeat(np.arange(2), pre.size(0)) 
        # tsne = TSNE(n_components=2, perplexity=int(math.sqrt(all_data.shape[0])), random_state=42)
        # X_2d = tsne.fit_transform(all_data)
        # plt.figure(figsize=(7, 5))
        # scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
        #                     c=y,               # 按类别着色
        #                     cmap='viridis',
        #                     s=25, alpha=0.8)
                            
        # for index in range(pre.size(0)):
        #     if index == 0:
        #         plt.arrow(X_2d[index, 0], X_2d[index, 1], X_2d[index + pre.size(0), 0] - X_2d[index, 0], X_2d[index + pre.size(0), 1] - X_2d[index, 1], head_width = 0.1, head_length = 0.1, fc = 'red', ec = 'red')
        #     else:
        #         plt.arrow(X_2d[index, 0], X_2d[index, 1], X_2d[index + pre.size(0), 0] - X_2d[index, 0], X_2d[index + pre.size(0), 1] - X_2d[index, 1], head_width = 0.1, head_length = 0.1, fc = 'black', ec = 'black')

        # plt.legend(scatter.legend_elements()[0], ['pre', 'post'], title='Class')
        # plt.xlabel('t-SNE dim 1')
        # plt.ylabel('t-SNE dim 2')
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(f'batch_trans_ot.png')
        # plt.close()

        # # torch.save(layer_out_ks, f'vector/{kwargs["index"]}_post.pt')

        # for q_ in test_question:
        #     q_tok = tok(
        #         q_,  
        #         return_tensors="pt",
        #         padding=True,
        #     ).to(model.device)
        #     print('----------------')
        #     print(tok.decode(model.generate(**q_tok, do_sample = False, max_new_tokens = 30)[0], skip_special_tokens = True).strip())
        #     print('----------------')
        # import ipdb; ipdb.set_trace()
        

        for x in [layer_in_ks, layer_out_ks,cur_zs, targets,stat_in,stat_out]:
            x.cpu()
            del x
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
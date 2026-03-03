import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from ...util import nethook
from ...util.globals import *
import torch.optim as optim
from ..rome.layer_stats import layer_stats
import argparse

import numpy as np
import os
from transformers.modeling_attn_mask_utils import AttentionMaskConverter,_prepare_4d_causal_attention_mask
from .memit_ARE_hparams import MEMITAREHyperParams
COV_CACHE = {}
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: MEMITAREHyperParams,
    layer: int,
    idxs_dict:dict,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(model.device)
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
    zs_out_list = []
    for k, idxs in idxs_dict.items():
        for idx in idxs:
            zs_out_list.append(zs_out[k,idx])
    zs_out = torch.stack(zs_out_list, dim=0)
    return zs_out


def generate_eval(prompt, edited_model, tok, hparams, max_new_tokens):
    question = tok([prompt], return_tensors='pt', padding=True)
    with torch.no_grad():
        generated_ids = edited_model.generate(
            input_ids=question['input_ids'].to(f'cuda:{hparams.device}'),
            attention_mask=question['attention_mask'].to(f'cuda:{hparams.device}'),
            do_sample = False,
            max_new_tokens = max_new_tokens
            # do_sample=True,
            # temperature=0.001,
            # max_new_tokens=512
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
    ]
    output = tok.batch_decode(generated_ids, skip_special_tokens=False)
    return output[0]

def apply_memit_ARE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data:list,
    hparams:MEMITAREHyperParams,
    copy: bool,
    ex_data:list,
    **kwargs
    ):


    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.sub_layers + hparams.target_layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}


    z_layer = hparams.sub_layers[-1]
    all_zs_list = []
    idxs_dict = {}
    for k, data in enumerate(batch_data[1:]):
        idxs_list, zs_list = compute_z(
            model,
            tok,
            data,
            z_layer,
            hparams
        )
        # 将当前样本的 target_list 中的向量添加到 all_target_vectors 中
        all_zs_list.extend(zs_list)
        idxs_dict[k] = idxs_list
    zs = torch.stack(all_zs_list, dim = 0)
    batch_question_ans = [
        i['prompt'] + i['target_new'] for i in batch_data[1:]
    ]
    


    sub_question = [d['prompt'] for d in batch_data[1:]]
    sub_answer = [d['target_new'] for d in batch_data[1:]]

    # Insert
    for i, layer in enumerate(hparams.sub_layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.rewrite_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        
        # wp change
        if i == 0:
            for key in idxs_dict:
                for index_ in range(len(idxs_dict[key])):
                    idxs_dict[key][index_] = idxs_dict[key][index_] + sum(contexts_tok['attention_mask'][key] == 0).item()

        cur_zs = compute_ks(model, tok,batch_question_ans, hparams, z_layer, idxs_dict)
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=1).mean())
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]

        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )

        ks_list = []
        kp_list = []
        for k, idxs in idxs_dict.items():
            all_idxs = set(range(len(layer_in_ks[k])))
            unselected_idxs = list(all_idxs - set(idxs))
            for idx in idxs:
                # 获取当前样本在指定索引下的layer_in_ks
                ks_list.append(layer_in_ks[k, idx])
            for unselected_idx in unselected_idxs:
                kp_list.append(layer_in_ks[k, unselected_idx])

        # 使用torch.stack将所有样本的layer_ks合并
        layer_ks = torch.stack(ks_list, dim=1)
        layer_kp = torch.stack(kp_list, dim=1)


        layer_kp, layer_ks, targets = (
            layer_kp.double(),
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov + layer_kp @ layer_kp.T +layer_ks @ layer_ks.T, #layer_kp @ layer_kp.T +
            layer_ks
        )
        resid = targets / (len(hparams.sub_layers) - i)  # Distribute residual across layers
        upd_matrix = resid.T @ adj_k.T
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks,layer_kp, cur_zs, targets, layer_in_ks, layer_out_ks]:
            x.cpu()
            del x
        torch.cuda.empty_cache()






    import ipdb; ipdb.set_trace()



    z_layer = hparams.target_layers[-1]
    all_zs_list = []
    idxs_dict = {}
    for k, data in enumerate([batch_data[0]]):
        idxs_list, zs_list = compute_z(
            model,
            tok,
            data,
            z_layer,
            hparams
        )
        # 将当前样本的 target_list 中的向量添加到 all_target_vectors 中
        all_zs_list.extend(zs_list)
        idxs_dict[k] = idxs_list
    zs = torch.stack(all_zs_list, dim = 0)
    batch_question_ans = [
        i['prompt'] + i['target_new'] for i in [batch_data[0]]
    ]
    
    # Insert
    for i, layer in enumerate(hparams.target_layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.rewrite_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        

        cur_zs = compute_ks(model, tok,batch_question_ans, hparams, z_layer, idxs_dict)
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=1).mean())
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]

        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )

        sub_data = [q + a for q, a in zip(sub_question, sub_answer)]

        sub_tok = tok(sub_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.rewrite_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**sub_tok)
                sub_stat_in = tr.input
                sub_stat_out = tr.output
        sub_stat_out = sub_stat_out[0] if type(sub_stat_out) is tuple else sub_stat_out

        ks_list = []
        kp_list = []
        for k, idxs in idxs_dict.items():
            all_idxs = set(range(len(layer_in_ks[k])))
            unselected_idxs = list(all_idxs - set(idxs))
            for idx in idxs:
                # 获取当前样本在指定索引下的layer_in_ks
                ks_list.append(layer_in_ks[k, idx])
            for unselected_idx in unselected_idxs:
                kp_list.append(layer_in_ks[k, unselected_idx])

        for index_ in range(len(sub_stat_out)):
            for index__ in range(len(sub_stat_out[0])):
                kp_list.append(sub_stat_in[index_][index__])

        # 使用torch.stack将所有样本的layer_ks合并
        layer_ks = torch.stack(ks_list, dim=1)
        layer_kp = torch.stack(kp_list, dim=1)

        layer_kp, layer_ks, targets = (
            layer_kp.double(),
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov + layer_kp @ layer_kp.T +layer_ks @ layer_ks.T, #layer_kp @ layer_kp.T +
            layer_ks
        )
        resid = targets / (len(hparams.target_layers) - i)  # Distribute residual across layers
        upd_matrix = resid.T @ adj_k.T
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks,layer_kp, cur_zs, targets, layer_in_ks, layer_out_ks]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    return model, weights_copy

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams = None
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dirs,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )

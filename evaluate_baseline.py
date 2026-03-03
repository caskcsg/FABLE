from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, QLoRAHyperParams, \
    GraceHyperParams, PMETHyperParams,MELOHyperParams, MALMENTrainingHparams, MALMENHyperParams, WISEHyperParams, R_ROMEHyperParams, EMMETHyperParams, \
    DeepEditApiHyperParams, DPOHyperParams, unkeHyperParams, MEMITAREHyperParams, fableHyperParams

from easyeditor import ZsreDataset, CounterFactDataset, KnowEditDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
import math
import random
import os
import json

import torch
from easyeditor.util import nethook
import argparse
from tqdm import tqdm
import numpy as np

from glue_eval.glue_eval import GLUEEval
from collections import defaultdict
from copy import deepcopy
import math
import glob


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

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

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', required=True, type=str)
parser.add_argument('--ds_size', default=10000, type=int)
parser.add_argument('--model_name', required=True, type=str)
parser.add_argument('--save_weights', required=True, type=str)
parser.add_argument('--device', required=True, type=str)
parser.add_argument('--question_template', default=0, type=int)

parser.add_argument('--total_machine', default=1, type=int)
parser.add_argument('--this_machine', default=0, type=int)

args = parser.parse_args()

set_seed(42)

ds_name = args.ds_name
model_name = args.model_name
device = args.device
save_weights = args.save_weights


if model_name == 'Llama3-8B-Instruct':
    hparams = FTHyperParams.from_hparams('./hparams/FT/llama3-8b.yaml')
elif model_name == 'Qwen2.5-7B-Instruct':
    hparams = FTHyperParams.from_hparams('./hparams/FT/qwen2.5-7b.yaml')



def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

hparams.model_name = f'../models/{model_name}'
hparams.question_template = args.question_template
hparams.device = device
hparams.stats_dirs = "./data/stats"

# hparams.fp16 = False
with open("./data/alpaca_data.json", 'r', encoding='utf-8') as f:
    ex_datas = json.load(f)

if hparams.question_template == 1:
    if model_name == 'Llama3-8B-Instruct':
        ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    elif model_name == 'Qwen2.5-7B-Instruct':
        ex_datas = [get_qwen_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
else:
    ex_datas = [i['instruction'] + i['input'] + i['output'] for i in ex_datas]

editor = BaseEditor.from_hparams(hparams)

ds_size = args.ds_size


if ds_name == 'cf':
    read_path = './data/AKEW/CounterFact.json'
elif ds_name == 'mquake':
    read_path = './data/AKEW/MQuAKE-CF.json'
elif ds_name == 'unke':
    read_path = './data/UnKE/final_data_v3.json'

raw_data = json.load(open(read_path, 'r'))[:args.ds_size]

target_new_threshold = 200
sub_answer_threshold = 30
target_new_tokens_num = []
sub_answer_tokens_num = []
for d in tqdm(raw_data):
    if ds_name == 'unke':
        target_new = [d['answer']]
    elif ds_name == 'mquake':
        target_new = [d["requested_rewrite"][0]["fact_new_uns"]]
    elif ds_name == 'cf':
        target_new = [d["requested_rewrite"]["fact_new_uns"]]

    sub_answer = []
    for item in d['sub_answer']:
        sub_answer.extend(item)

    target_new_tokens_num.extend([len(editor.tok.encode(q)) for q in target_new])
    sub_answer_tokens_num.extend([len(editor.tok.encode(q)) for q in sub_answer])

print(sum(np.array(target_new_tokens_num) <= target_new_threshold) / len(target_new_tokens_num))
print(sum(np.array(sub_answer_tokens_num) <= sub_answer_threshold) / len(sub_answer_tokens_num))

save_post = f'./results/{ds_name}/{model_name}/pre'
os.makedirs(save_post, exist_ok = True)


weights_done = glob.glob(save_post + '/*.json')
done_index = []
all_index = [i for i in range(len(raw_data))]
for path in weights_done:
    index = int(path.split('/')[-1].split('.')[0])
    done_index.append(index)
todo_index = list(set(all_index) - set(done_index))
todo_index.sort()
raw_data_todo = []
for index in todo_index:
    raw_data_todo.append(raw_data[index])


total_machine = int(args.total_machine)
this_machine = int(args.this_machine)
chunk = math.ceil(len(raw_data_todo) / total_machine)
start = this_machine * chunk
end = start + chunk if start + chunk < len(raw_data_todo) else len(raw_data_todo)

weights_copy = {}

ex_data_list = []
for i in range(2000):
    ex_data_list.append(random.sample(ex_datas, 20))

sample = 5
glue_eval = GLUEEval(editor.model, editor.tok)

random.shuffle(glue_eval.sst_eval.eval_dataset)
random.shuffle(glue_eval.mmlu_eval.eval_dataset)
random.shuffle(glue_eval.mrpc_eval.eval_dataset)
random.shuffle(glue_eval.cola_eval.eval_dataset)
random.shuffle(glue_eval.rte_eval.eval_dataset)
random.shuffle(glue_eval.sentiment_analysis_eval.eval_dataset)
random.shuffle(glue_eval.nli_eval.eval_dataset)
random.shuffle(glue_eval.dialogue_eval.eval_dataset)

index = start

sst_eval_dataset = deepcopy(glue_eval.sst_eval.eval_dataset) * (2000 // (len(glue_eval.sst_eval.eval_dataset) // sample))
mmlu_eval_dataset = deepcopy(glue_eval.mmlu_eval.eval_dataset) * (2000 // (len(glue_eval.mmlu_eval.eval_dataset) // sample))
mrpc_eval_dataset = deepcopy(glue_eval.mrpc_eval.eval_dataset) * (2000 // (len(glue_eval.mrpc_eval.eval_dataset) // sample))
cola_eval_dataset = deepcopy(glue_eval.cola_eval.eval_dataset) * (2000 // (len(glue_eval.cola_eval.eval_dataset) // sample))
rte_eval_dataset = deepcopy(glue_eval.rte_eval.eval_dataset) * (2000 // (len(glue_eval.rte_eval.eval_dataset) // sample))
sentiment_analysis_eval_dataset = deepcopy(glue_eval.sentiment_analysis_eval.eval_dataset) * (2000 // (len(glue_eval.sentiment_analysis_eval.eval_dataset) // sample))
nli_eval_dataset = deepcopy(glue_eval.nli_eval.eval_dataset) * (2000 // (len(glue_eval.nli_eval.eval_dataset) // sample))
dialogue_eval_dataset = deepcopy(glue_eval.dialogue_eval.eval_dataset) * (2000 // (len(glue_eval.dialogue_eval.eval_dataset) // sample))


for d, index in tqdm(zip(raw_data_todo[start:end], todo_index[start:end]), desc = f'{start}-{end}'):
    print('index', index)
    
    if os.path.exists(save_post + f'/{index}.json'):
        # index += 1
        # continue
        t_dict = json.load(open(save_post + f'/{index}.json', 'r'))
    else:
        t_dict = {}

    set_seed(42)

    if ds_name == 'cf':
        prompts = [d["requested_rewrite"]["question"]]
        paraphrase = d['paraphrase_prompts']
        sub_question = d['sub_question']
    elif ds_name == 'mquake':
        prompts = [d["requested_rewrite"][0]["question"]]
        paraphrase = [None]
        sub_question = d['sub_question']
    elif ds_name == 'unke':
        prompts = [d['question']] if isinstance(d['question'], str) else d['question']
        paraphrase = [d['para_question']] if isinstance(d['para_question'], str) else d['para_question']
        sub_question = d['sub_question']


    if hparams.question_template == 1:
        if model_name == 'Llama3-8B-Instruct':
            prompts = [get_llama_without_answer(q) for q in prompts]
            paraphrase = [get_llama_without_answer(q) if q != None else None  for q in paraphrase]
            sub_question = [get_llama_without_answer(q) for q in sub_question]
        elif model_name == 'Qwen2.5-7B-Instruct':
            prompts = [get_qwen_without_answer(q) for q in prompts]
            paraphrase = [get_qwen_without_answer(q) if q != None else None for q in paraphrase]
            sub_question = [get_qwen_without_answer(q) for q in sub_question]


    for key in d:
        if key not in t_dict:
            t_dict[key] = d[key]
    
    if 'post_eval' not in t_dict:
        t_dict['post_eval'] = {}

    if 'rewrite_output' not in t_dict['post_eval']:
        rewrite_output = [generate_eval(p, editor.model, editor.tok, hparams, target_new_threshold) for p in prompts]
        t_dict['post_eval']['rewrite_output'] = rewrite_output
    
    if 'sub_question_output' not in t_dict['post_eval']:
        sub_question_output = [generate_eval(q, editor.model, editor.tok, hparams, sub_answer_threshold) if q != None else None for q in sub_question]
        t_dict['post_eval']['sub_question_output'] = sub_question_output


    if 'downstream_eval' not in t_dict['post_eval']:
        glue_eval.sst_eval.eval_dataset = sst_eval_dataset[index * sample: index * sample + sample]
        glue_eval.mmlu_eval.eval_dataset = mmlu_eval_dataset[index * sample: index * sample + sample]
        glue_eval.mrpc_eval.eval_dataset = mrpc_eval_dataset[index * sample: index * sample + sample]
        glue_eval.cola_eval.eval_dataset = cola_eval_dataset[index * sample: index * sample + sample]
        glue_eval.rte_eval.eval_dataset = rte_eval_dataset[index * sample: index * sample + sample]
        glue_eval.sentiment_analysis_eval.eval_dataset = sentiment_analysis_eval_dataset[index * sample: index * sample + sample]
        glue_eval.nli_eval.eval_dataset = nli_eval_dataset[index * sample: index * sample + sample]
        glue_eval.dialogue_eval.eval_dataset = dialogue_eval_dataset[index * sample: index * sample + sample]
        glue_results = glue_eval.evaluate(defaultdict(dict), nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True, gen_len = 20)
        t_dict['post_eval']['downstream_eval'] = glue_results

        
    json.dump(t_dict, open(save_post + f'/{index}.json', 'w'), indent=4, ensure_ascii=False)
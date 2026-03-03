import json
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse
import sys
import os
sys.setrecursionlimit(2000)

from easyeditor.evaluate.evaluate_utils import n_gram_entropy
import nltk
import glob
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('--post_path', required=True, type=str)
parser.add_argument('--model_path', default='sentence-transformers/all-MiniLM-L6-v2', type=str)
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()



def longest_subsequence_substring(short: str, long: str) -> str:
    """
    找出 short 中能在 long 中按顺序匹配的最长非连续子串。
    返回这个子串（保留 short 中的原始顺序和单词）。
    """
    if not short.strip():
        return ""

    # 正则分词，统一小写
    short_words = re.findall(r'[A-Za-z0-9]+', short)
    long_words = re.findall(r'[A-Za-z0-9]+', long)

    # 动态规划找最长公共子序列（LCS）
    m, n = len(short_words), len(long_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 构建 dp 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if short_words[i - 1].lower() == long_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯找出最长子序列
    i, j = m, n
    lcs_words = []
    while i > 0 and j > 0:
        if short_words[i - 1].lower() == long_words[j - 1]:
            lcs_words.append(short_words[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_words.reverse()

    return (len(lcs_words) == len(short_words), len(lcs_words) / len(short_words))

post_data = []

post_data_files = glob.glob(args.post_path + '/*.json')
post_data_files.sort(key = lambda x : int(x.split('/')[-1].replace('.json', '')))

for path in post_data_files:
    post_data.append(json.load(open(path, 'r')))

ds_name = args.post_path.split('/')[-5]


metrics = {}

# cal bleu
bleu_scores = []
bleu_scores_sub = []

rouge1s=[]
rouge2s=[]
rougels=[]
rougels_sub=[]
rouge1s_sub=[]
rouge2s_sub=[]
rouge = Rouge()

hit_rate_ori=[]

sentences1 = []
sentences2 = []
sentences3 = []
sentences4 = []
sentences5 = []
sentences5_num = []

downstream_scores = {
    'sst': [], 
    'mmmlu': [], 
    'mrpc': [], 
    'cola': [], 
    'rte': [], 
    'nli': []
}

index = 0
for post in tqdm(post_data):
    if ds_name == 'unke':
        answer = post['answer']
    elif ds_name == 'mquake':
        answer = post['requested_rewrite'][0]['fact_new_uns']
    elif ds_name == 'cf':
        answer = post['requested_rewrite']['fact_new_uns']

    sub_answer = post['sub_answer']
    sub_answer_distill = post['sub_answer_distill']

    post_rewrite_output = [a.replace('<|eot_id|>', '').replace('<|im_end|>', '').strip() for a in post['post_eval']['rewrite_output']]
    post_sub_question_output = [a.replace('<|eot_id|>', '').replace('<|im_end|>', '').strip() for a in post['post_eval']['sub_question_output']]


    post_rewrite_output = [a if ' ' in a else ' ' + a for a in post_rewrite_output]
    post_sub_question_output = [a if ' ' in a else ' ' + a for a in post_sub_question_output]

    for key, value in downstream_scores.items():
        downstream_scores[key].append(post['post_eval']['downstream_eval'][key]['score']['f1'])

    sentences1.append(answer)
    sentences2.extend(post_rewrite_output)
    sentences3.extend(sub_answer_distill)
    sentences4.extend(sub_answer)
    sentences5.extend(post_sub_question_output)

    sentences5_num.append(len(post_sub_question_output))

    score = sentence_bleu([answer], post_rewrite_output[0])
    bleu_scores.append(score)

    sub_score = 0
    for i in range(len(post_sub_question_output)):
        sub_score += sentence_bleu([sub_answer[i]], post_sub_question_output[i])
    bleu_scores_sub.append(sub_score/len(post_sub_question_output))

    scores = rouge.get_scores(post_rewrite_output[0], answer)
    rouge1s.append(scores[0]['rouge-1']['r'])
    rouge2s.append(scores[0]['rouge-2']['r'])
    rougels.append(scores[0]['rouge-l']['r'])


    sub_ls = 0
    sub_1s = 0
    sub_2s = 0
    for i in range(len(post_sub_question_output)):
        scores = rouge.get_scores(post_sub_question_output[i], sub_answer[i])
        sub_1s += scores[0]['rouge-1']['r']
        sub_2s += scores[0]['rouge-2']['r']
        sub_ls += scores[0]['rouge-l']['r']
    rouge1s_sub.append(sub_1s/len(post_sub_question_output))
    rouge2s_sub.append(sub_2s/len(post_sub_question_output))
    rougels_sub.append(sub_ls/len(post_sub_question_output))

    index += 1


temp_original = {}
temp_sub={}
temp_downstream = {}

# cal bert score
print("***********Calculate BERT Similarity Score**************")
model = SentenceTransformer(args.model_path, device=f"cuda:{args.device}")

embeddings1 = model.encode(sentences1, convert_to_tensor=True,show_progress_bar=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True,show_progress_bar=True)

embeddings6 = model.encode(sentences4, convert_to_tensor=True,show_progress_bar=True)
embeddings7 = model.encode(sentences5, convert_to_tensor=True,show_progress_bar=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)
# print(cosine_scores.shape)


temp_original['Bert Score'] = (np.round(100 * np.mean(cosine_scores.diagonal().cpu().tolist()), 2), np.round(np.std(cosine_scores.diagonal().cpu().tolist()), 2))


cosine_scores = util.cos_sim(embeddings6, embeddings7)
start = 0
avg_score = []
for num in sentences5_num:
    end = start + num
    avg_score.append(cosine_scores[start:end, start:end].diagonal().mean().item())
    start = end

temp_sub['Bert Score'] = (np.round(100 * np.mean(avg_score), 2), np.round(np.std(avg_score), 2))

temp_original['ROUGE-L'] = (np.round(100 * np.mean(rougels), 2), np.round(np.std(rougels), 2))

temp_sub['ROUGE-L'] = (np.round(100 * np.mean(rougels_sub), 2), np.round(np.std(rougels_sub), 2))

# Downstream eval
for key, value in downstream_scores.items():
    temp_downstream[key] = (np.round(100 * np.mean(value), 2), np.round(np.std(value), 2))


# # 专为Sub设计一系列指标

start = 0
hit_rate = []
longgest_hit_rate = []


index = 0
for num in sentences5_num:
    end = start + num
    hit_num = 0
    hit_total = 0
    longgest_hit_num = 0

    for a, pred, a_distill in zip(sentences4[start:end], sentences5[start:end], sentences3[start:end]):
        a = a[:-1] if '.' in a[-1] else a
        a = a.lower().strip()
        pred = pred[:-1] if '.' in pred[-1] else pred
        pred = pred.lower().strip()

        a_distill = [a_[:-1] if '.' in a_[-1] else a_ for a_ in a_distill]
        a_distill = [a_.lower().strip() for a_ in a_distill]



        for a_ in a_distill:
            hit_num += int(longest_subsequence_substring(a_, pred)[0])
            hit_total += 1
        
        if a != '':
            longgest_hit_num += longest_subsequence_substring(a, pred)[1]
        else:
            import ipdb; ipdb.set_trace()

    hit_rate.append(hit_num / hit_total)
    longgest_hit_rate.append(longgest_hit_num / num)

    start = end

    index += 1

temp_sub['Hit Rate'] = (np.round(100 * np.mean(hit_rate), 2), np.round(np.std(hit_rate), 2))
temp_sub['Longgest Hit Rate'] = (np.round(100 * np.mean(longgest_hit_rate), 2), np.round(np.std(longgest_hit_rate), 2))



metrics['Original']=temp_original
metrics['Sub']=temp_sub
metrics['Downstream'] = temp_downstream

print("***********Result**************")
print(args.post_path)
print(temp_original)
print(temp_sub)
print(temp_downstream)

json.dump(metrics, open(args.post_path.replace('/post', '').replace('/pre', '') + '/results.json', 'w'), indent = 4, ensure_ascii = False)
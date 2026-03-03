import glob
import json
from collections import defaultdict
import pandas as pd

all_files = glob.glob('/public/wp/FineUnstruct/results/*/*/*/*/results.json')
all_files.extend(glob.glob('/public/wp/FineUnstruct/results/*/*/results.json'))

col_names = ['Model', 'Method', 'Args',
                'Original|Bert Score', 'Original|ROUGE-L', 
                'Para|Bert Score', 'Para|ROUGE-L', 
                'Sub|Bert Score', 'Sub|ROUGE-L', 'Sub|Hit Rate', 'Sub|Longgest Hit Rate', 'Sub|Fluency',
                'Downstream|sst', 'Downstream|mmmlu', 'Downstream|mrpc', 'Downstream|cola', 'Downstream|rte', 'Downstream|nli'
            ]
col_names_short = ['Model', 'Method', 'Args',
                'Ori.Bert', 'Ori.ROUGE', 
                'Para.Bert', 'Para.ROUGE', 
                'Sub.Bert', 'Sub.ROUGE', 'Sub.Hit', 'Sub.Longgest', 'Sub.Fluency',
                'sst', 'mmmlu', 'mrpc', 'cola', 'rte', 'nli'
            ]

ds2row = defaultdict(list)
for path in all_files:
    path_split = path.split('/')
    if path_split[-4] == 'results':
        ds_name, model_name = path_split[-3:-1]
        method_name, args = 'Pre', '-'
    else:
        ds_name, model_name, method_name, args = path_split[-5:-1]
    data = json.load(open(path, 'r'))

    score_list = []
    for col in col_names[3:]:
        score = data[col.split('|')[0]][col.split('|')[1]]
        avg, std = score[0], score[1]
        if col.split('|')[0] != 'Downstream':
            score_list.append(f'{avg:.2f}±{std:.2f}')
        else:
            score_list.append(f'{avg:.2f}')
            
    ds2row[ds_name].append([model_name, method_name, args] + score_list)

order_model = {'Llama3-8B-Instruct': 0, 'Qwen2.5-7B-Instruct': 1}
order_method = {'Pre': 0, 'FT-L': 1, 'FT-M': 2, 'ROME': 3, 'MEMIT': 4, 'MEMIT_ARE': 5, 'UnKE': 6, 'UnKE_Debug': 7, '_UnKE_Debug': 8}
for ds_name in ds2row:
    rows = ds2row[ds_name]
    rows.sort(key = lambda x : (order_model[x[0]], order_method[x[1]]))
    save_data = pd.DataFrame(rows, columns = col_names_short)
    save_path = f'results/{ds_name}.csv'
    save_data.to_csv(save_path, index = False, encoding='utf-8-sig')
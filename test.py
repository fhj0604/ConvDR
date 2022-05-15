import subprocess as sp
import os
import time 
import pandas as pd
import json
import torch
# PATH = 'tmp2/results/or-quac/multi_task.jsonl'
# PATH = 'tmp2/results/or-quac/multi_task_answer.jsonl'
# PATH = 'tmp2/results/or-quac/multi_task_answer.trec'
# PATH = 'tmp2/results/or-quac/manual_ance_dev.jsonl'
# PATH = 'tmp2/results/or-quac/manual_ance_dev.trec'
# PATH = 'tmp2/datasets/raw/or-quac/preprocessed/dev.txt'
# PATH2 = './tmp2/datasets/or-quac/dev.rerank.jsonl'
# PATH = './tmp2/datasets/or-quac/queries.dev.manual.tsv'
# PATH = './tmp2/datasets/or-quac/collection.jsonl'
PATH = './tmp2/datasets/or-quac/collection_rerank.jsonl'
PATH2 = './tmp2/datasets/or-quac/collection_rerank_2.jsonl'
# PATH = './tmp2/datasets/or-quac/qrels.tsv'
PATH = './tmp2/datasets/or-quac/dev.rerank.jsonl'


with open(PATH, 'r') as f:
    for line in f.readlines():
        dic = json.loads(line)
        print(dic)
        print(dic.keys())
        break
# with open(PATH, 'r') as f, open(PATH2, 'w') as g:
#     for line in f.readlines():
#         dic = json.loads(line)
#         out_obj = {
#             "doc_id": str(dic["doc_id"]),
#             "doc": dic["doc"]
#         }
#         out_line = json.dumps(out_obj) + '\n'
#         g.write(out_line)
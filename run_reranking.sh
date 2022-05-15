# python ./drivers/run_reranking.py --train_file ./results/cast-19/multi.jsonl 

if [[ $2 = "c19" ]]
    then
    data_file=./tmp2/datasets/cast-19/collection.tsv
    result_dir=./tmp2/results/cast-19
elif [[ $2 = "or" ]]
    then
    data_file=./tmp2/datasets/or-quac/collection_rerank.jsonl
    result_dir=./tmp2/results/or-quac
    data_dir=./tmp2/datasets/or-quac
fi

if [[ $1 = "train" ]]
	then
    # echo "generating rerank training data"
    # python data/gen_rerank_data.py --data_path ${data_dir}/train.rank.jsonl --output_path ${data_dir}/train.rerank.jsonl
    # echo "generating rerank dev data"
    # python data/gen_rerank_data.py --data_path ${data_dir}/dev.rank.jsonl --output_path ${data_dir}/dev.rerank.jsonl
    # -train queries=${result_dir}/manual_ance_train.jsonl,docs=${data_file},trec=${result_dir}/manual_ance_train.trec,qrels=${data_dir}/qrels.tsv \
    echo "run reranking"
	CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 reranker/train.py \
        -task ranking \
        -model bert \
        -train ${data_dir}/train.rerank.jsonl \
        -max_input 12800000 \
        -save ./tmp2/checkpoints/reranker/bert.bin \
        -dev queries=${result_dir}/manual_ance_dev.jsonl,docs=${data_file},trec=${result_dir}/manual_ance_dev.trec,qrels=${data_dir}/qrels.tsv \
        -qrels ./data/qrels.dev.small.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./tmp2/checkpoints/MSMARCO/bert-base.bin \
        -res ./results/bert.trec \
        -metric mrr_cut_10 \
        -max_query_len 32 \
        -max_doc_len 221 \
        -epoch 1 \
        -batch_size 2 \
        -gradient_accumulation_steps 10 \
        -lr 3e-6 \
        -n_warmup_steps 160000 \
        -eval_every 10000 \

elif [[ $1 = "inference" ]]
	then    
	python reranker/inference.py \
		-task ranking \
		-model bert \
		-max_input 12800000 \
		-test queries=${result_dir}/$3.jsonl,docs=${data_file},trec=${result_dir}/$3.trec,qrels=${data_dir}/qrels.tsv \
		-vocab bert-base-uncased \
		-pretrain bert-base-uncased \
		-checkpoint ./tmp2/checkpoints/MSMARCO/bert-base.bin \
		-res ${result_dir}/$3_rerank.trec \
		-max_query_len 32 \
		-max_doc_len 221 \
		-batch_size 256
fi
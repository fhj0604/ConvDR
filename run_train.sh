if [[ $1 = "c19" ]]
    then 
    if [[ $2 = "kd" ]]
        then 
        echo "Running CAsT-19 with kd loss"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-kd-cast19  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --train_file=tmp2/datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate 
    elif [[ $2 = "neg" ]]
        then
        echo "Running CAsT-19, find negative documents for each query"
        python drivers/run_convdr_inference.py  --model_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --eval_file=tmp2/datasets/cast-19/eval_topics.jsonl  --query=target  --per_gpu_eval_batch_size=1  --ann_data_dir=tmp2/datasets/cast-shared/embeddings  --qrels=tmp2/datasets/cast-19/qrels.tsv  --processed_data_dir=tmp2/datasets/cast-shared/tokenized  --raw_data_dir=tmp2/datasets/cast-19   --output_file=tmp2/results/cast-19/manual_ance.jsonl  --output_trec_file=tmp2/results/cast-19/manual_ance.trec  --model_type=rdot_nll  --output_query_type=manual  --use_gpu
    elif [[ $2 = "gen" ]]
        then 
        echo "Running CAsT-19, run gen_ranking_data.py"
        python data/gen_ranking_data.py  --train=tmp2/datasets/cast-19/eval_topics.jsonl  --run=tmp2/results/cast-19/manual_ance.trec  --output=tmp2/datasets/cast-19/eval_topics.rank.jsonl  --qrels=tmp2/datasets/cast-19/qrels.tsv  --collection=tmp2/datasets/cast-19/collection.tsv  --cast
    elif [[ $2 = "multi" ]]
        then 
        echo "Running CAsT-19 with multi loss"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-multi-cast19  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --train_file=tmp2/datasets/cast-19/eval_topics.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_multi_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --ranking_task
    elif [[ $2 = "all" ]]
        then 
        echo "Running CAsT-19, find negative documents for each query"
        python drivers/run_convdr_inference.py  --model_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --eval_file=tmp2/datasets/cast-19/eval_topics.jsonl  --query=target  --per_gpu_eval_batch_size=1  --ann_data_dir=tmp2/datasets/cast-shared/embeddings  --qrels=tmp2/datasets/cast-19/qrels.tsv  --processed_data_dir=tmp2/datasets/cast-shared/tokenized  --raw_data_dir=tmp2/datasets/cast-19   --output_file=tmp2/results/cast-19/manual_ance.jsonl  --output_trec_file=tmp2/results/cast-19/manual_ance.trec  --model_type=rdot_nll  --output_query_type=manual  --use_gpu
        echo "Running CAsT-19, run gen_ranking_data.py"
        python data/gen_ranking_data.py  --train=tmp2/datasets/cast-19/eval_topics.jsonl  --run=tmp2/results/cast-19/manual_ance.trec  --output=tmp2/datasets/cast-19/eval_topics.rank.jsonl  --qrels=tmp2/datasets/cast-19/qrels.tsv  --collection=tmp2/datasets/cast-19/collection.tsv  --cast
        echo "Running CAsT-19 with multi loss"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-multi-cast19  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --train_file=tmp2/datasets/cast-19/eval_topics.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_multi_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --ranking_task
    fi

elif [[ $1 = "or" ]]
    then 
    if [[ $2 = "kd" ]]
        then 
        echo "Running or-quac with kd loss"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-kd-orquac.cp  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --train_file=tmp2/datasets/or-quac/train.jsonl  --query=no_res  --per_gpu_train_batch_size=1  --learning_rate=1e-5  --log_dir=logs/convdr_kd_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100 --gradient_accumulation_steps=10
    elif [[ $2 = "kd-answer" ]]
        then
        echo "Running or-quac with kd loss (add answer)"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-kd-answer-orquac.cp  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --train_file=tmp2/datasets/or-quac/train.jsonl  --query=no_res  --per_gpu_train_batch_size=1  --learning_rate=1e-5  --log_dir=logs/convdr_kd_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100 --add_answer --gradient_accumulation_steps=10
        
    elif [[ $2 = "neg" ]]
        then
        echo "Running or-quac, find negative documents for each query"
        python drivers/run_convdr_inference.py  --model_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --eval_file=tmp2/datasets/or-quac/train.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=tmp2/datasets/or-quac/embeddings  --qrels=tmp2/datasets/or-quac/qrels.tsv  --processed_data_dir=tmp2/datasets/or-quac/tokenized  --raw_data_dir=tmp2/datasets/or-quac   --output_file=tmp2/results/or-quac/manual_ance_train.jsonl  --output_trec_file=tmp2/results/or-quac/manual_ance_train.trec  --model_type=dpr  --output_query_type=train.manual  --use_gpu
    elif [[ $2 = "gen" ]]
        then
        echo "Running or-quac, run gen_ranking_data.py"
        python data/gen_ranking_data.py  --train=tmp2/datasets/or-quac/train.jsonl  --run=tmp2/results/or-quac/manual_ance_train.trec  --output=tmp2/datasets/or-quac/train.rank.jsonl  --qrels=tmp2/datasets/or-quac/qrels.tsv  --collection=tmp2/datasets/or-quac/collection.jsonl
    elif [[ $2 = "multi" ]]
        then 
        echo "Running or-quac with multi loss"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-multi-orquac.cp  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --train_file=tmp2/datasets/or-quac/train.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=1  --learning_rate=1e-5  --log_dir=logs/convdr_multi_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task --gradient_accumulation_steps=10
    elif [[ $2 = "multi-answer" ]]
        then 
        echo "Running or-quac with multi loss (add answer)"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-multi-answer-orquac.cp  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --train_file=tmp2/datasets/or-quac/train.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=1 --learning_rate=1e-5  --log_dir=logs/convdr_multi_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task --add_answer --gradient_accumulation_steps=10
    elif [[ $2 = "all" ]] 
        then
        echo "Running or-quac, find negative documents for each query"
        python drivers/run_convdr_inference.py  --model_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --eval_file=tmp2/datasets/or-quac/train.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=tmp2/datasets/or-quac/embeddings  --qrels=tmp2/datasets/or-quac/qrels.tsv  --processed_data_dir=tmp2/datasets/or-quac/tokenized  --raw_data_dir=tmp2/datasets/or-quac   --output_file=tmp2/results/or-quac/manual_ance_train.jsonl  --output_trec_file=tmp2/results/or-quac/manual_ance_train.trec  --model_type=dpr  --output_query_type=train.manual  --use_gpu
        echo "Running or-quac, run gen_ranking_data.py"
        python data/gen_ranking_data.py  --train=tmp2/datasets/or-quac/train.jsonl  --run=tmp2/results/or-quac/manual_ance_train.trec  --output=tmp2/datasets/or-quac/train.rank.jsonl  --qrels=tmp2/datasets/or-quac/qrels.tsv  --collection=tmp2/datasets/or-quac/collection.jsonl
        echo "Running or-quac with multi loss"
        python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-multi-orquac.cp  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --train_file=tmp2/datasets/or-quac/train.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5  --log_dir=logs/convdr_multi_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task
    fi
fi  
        






# CAsT-19, KD loss only, five-fold cross-validation
# python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-kd-cast19  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --train_file=tmp2/datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate
# CAsT-20, KD loss only, five-fold cross-validation, use automatic canonical responses, set a longer length
# python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-kd-cast20  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --train_file=tmp2/datasets/cast-20/eval_topics.jsonl  --query=auto_can  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast20  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --max_concat_length=512
# OR-QuAC, KD loss only
# python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-kd-orquac.cp  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --train_file=tmp2/datasets/or-quac/train.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5  --log_dir=logs/convdr_kd_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100


# CAsT-19
# python drivers/run_convdr_inference.py  --model_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --eval_file=tmp2/datasets/cast-19/eval_topics.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=tmp2/datasets/cast-shared/embeddings  --qrels=tmp2/datasets/cast-19/qrels.tsv  --processed_data_dir=tmp2/datasets/cast-shared/tokenized  --raw_data_dir=tmp2/datasets/cast-19   --output_file=results/cast-19/manual_ance.jsonl  --output_trec_file=results/cast-19/manual_ance.trec  --model_type=rdot_nll  --output_query_type=manual  --use_gpu
# OR-QuAC, inference on train, set query to "target" to use manual queries directly
# python drivers/run_convdr_inference.py  --model_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --eval_file=tmp2/datasets/or-quac/train.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=tmp2/datasets/or-quac/embeddings  --qrels=tmp2/datasets/or-quac/qrels.tsv  --processed_data_dir=tmp2/datasets/or-quac/tokenized  --raw_data_dir=tmp2/datasets/or-quac   --output_file=results/or-quac/manual_ance_train.jsonl  --output_trec_file=results/or-quac/manual_ance_train.trec  --model_type=dpr  --output_query_type=train.manual  --use_gpu


# CAsT-19
# python data/gen_ranking_data.py  --train=tmp2/datasets/cast-19/eval_topics.jsonl  --run=results/cast-19/manual_ance.trec  --output=tmp2/datasets/cast-19/eval_topics.rank.jsonl  --qrels=tmp2/datasets/cast-19/qrels.tsv  --collection=tmp2/datasets/cast-19/collection.tsv  --cast
# OR-QuAC
# python data/gen_ranking_data.py  --train=datasets/or-quac/train.jsonl  --run=results/or-quac/manual_ance_train.trec  --output=datasets/or-quac/train.rank.jsonl  --qrels=datasets/or-quac/qrels.tsv  --collection=datasets/or-quac/collection.jsonl

# CAsT-19, Multi-task
# python drivers/run_convdr_train.py  --output_dir=tmp2/checkpoints/convdr-multi-cast19  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco  --train_file=tmp2/datasets/cast-19/eval_topics.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_multi_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --ranking_task
# OR-QuAC, Multi-task
# python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-multi-orquac.cp  --model_name_or_path=checkpoints/ad-hoc-ance-orquac.cp  --train_file=datasets/or-quac/train.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5  --log_dir=logs/convdr_multi_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task
if [[ $1 = "c19" ]]
    then
    if [[ $2 = "kd" ]]
    then 
        echo "trec evaluating cast-19 with kd loss"
        ./trec_eval -m all_trec tmp2/datasets/cast-19/qrels.tsv results/cast-19/kd.trec
    elif [[ $2 = "multi" ]]
    then 
        echo "trec evaluating cast-19 with multi loss"
        ./trec_eval -m all_trec tmp2/datasets/cast-19/qrels.tsv results/cast-19/multi_rerank.trec
    elif [[ $2 = "manual" ]]
    then 
        echo "trec evaluating cast-19 with manual query" 
        ./trec_eval -m all_trec tmp2/datasets/cast-19/qrels.tsv results/cast-19/manual_ance.trec
    fi
elif [[ $1 = "or" ]]
    then
    if [[ $2 = "kd" ]]
    then 
        echo "trec evaluating or-quac with kd loss"
        ./trec_eval -m all_trec tmp2/datasets/or-quac/qrels.tsv tmp2/results/or-quac/kd.trec
    elif [[ $2 = "multi" ]]
    then 
        echo "trec evaluating or-quac with multi loss"
        ./trec_eval -m all_trec tmp2/datasets/or-quac/qrels.tsv tmp2/results/or-quac/multi_task.trec
    elif [[ $2 = "multi-answer" ]]
    then 
        echo "trec evaluating or-quac with multi loss (answer)"
        ./trec_eval -m all_trec tmp2/datasets/or-quac/qrels.tsv tmp2/results/or-quac/multi_task_answer.trec
    elif [[ $2 = "manual" ]]
    then 
        echo "trec evaluatin or-quac with manual query"
        ./trec_eval -m all_trec tmp2/datasets/or-quac/qrels.tsv results/or-quac/manual_ance_train.trec
    fi
fi

# ./trec_eval tmp2/datasets/cast-19/qrels.tsv results




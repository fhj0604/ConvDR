#=====routes=====
base="/home/fhj0604/ConvDR"
ckpt=${base}"/tmp2/checkpoints"
datasets=${base}"/tmp2/datasets"
raw=${base}"/tmp2/datasets/raw"
cast_19_raw=${raw}"/cast-19"
cast_20_raw=${raw}"/cast-20"
cast_shared=${datasets}"/cast-shared"
cast_19=${datasets}"/cast-19"
cast_20=${datasets}"/cast-20"
or_quac=${datasets}"/or-quac"
or_quac_raw=${raw}"/or-quac"
tokenized_shared=${cast_shared}"/tokenized"
embeddings_shared=${cast_shared}"/embeddings"
tokenized_orquac=${or_quac}"/tokenized"
embeddings_orquac=${or_quac}"/embeddings"




#=====create directory=====
if [ ! -d ${ckpt} ]; then
    mkdir ${ckpt}
fi

if [ ! -d ${datasets} ]; then
    mkdir ${datasets}
    mkdir ${raw}
    mkdir ${cast_19_raw}
    mkdir ${cast_20_raw}
fi


# mkdir ${cast_19}
# mkdir ${cast_20}
# mkdir ${cast_shared}
# mkdir ${or_quac_raw}
# mkdir ${or_quac}

#=====download file=====
if [ ! -d ${ckpt}/ad-hoc-ance-msmarco ] || [ ! -f ${ckpt}/ad-hoc-ance-orquac.cp ]; then
    cd ${ckpt}
    if [ ! -f ad-hoc-ance-orquac.cp ]; then
        wget https://data.thunlp.org/convdr/ad-hoc-ance-orquac.cp
    fi
    if [ ! -d ad-hoc-ance-msmarco ]; then 
        wget https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip
        unzip Passage_ANCE_FirstP_Checkpoint.zip
        mv "Passage ANCE(FirstP) Checkpoint" ad-hoc-ance-msmarco
    fi
fi 

if [ ! -f ${raw}/msmarco.tsv ] || [ ! -f ${raw}/paragraphCorpus.v2.0.tar.xz ] || [ ! -f ${raw}/duplicate_list_v1.0.txt ]; then
    cd ${raw}
    if [ ! -f msmarco.tsv ]; then
        wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -O msmarco.tsv
    fi
    if [ ! -f ${raw}/paragraphCorpus.v2.0.tar.xz ]; then
        wget http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
        tar Jxvf paragraphCorpus.v2.0.tar.xz
    fi
    if [ ! -f ${raw}/paragraphCorpus.v2.0.tar.xz ]; then
        wget http://boston.lti.cs.cmu.edu/Services/treccast19/duplicate_list_v1.0.txt
    fi
fi

if [ ! -f ${cast_19_raw}/evaluation_topics_v1.0.json ] || [ ! -f ${cast_19_raw}/evaluation_topics_annotated_resolved_v1.0.tsv ] || [ ! -f ${cast_19_raw}/2019qrels.txt ]; then
    cd ${cast_19_raw}
    if [ ! -f ${cast_19_raw}/evaluation_topics_v1.0.json ]; then 
        wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json
    fi
    if [ ! -f ${cast_19_raw}/evaluation_topics_annotated_resolved_v1.0.tsv ]; then
        wget wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
    fi
    if [ ! -f ${cast_19_raw}/2019qrels.txt ]; then
        wget https://trec.nist.gov/data/cast/2019qrels.txt
    fi
fi

if [ ! -f ${cast_20_raw}/2020_automatic_evaluation_topics_v1.0.json ] || [ ! -f ${cast_20_raw}/2020_manual_evaluation_topics_v1.0.json ] || [ ! -f ${cast_20_raw}/2020qrels.txt ]; then
    cd ${cast_20_raw}
    if [ ! -f ${cast_20_raw}/2020_automatic_evaluation_topics_v1.0.json ]; then 
        wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_automatic_evaluation_topics_v1.0.json
    fi
    if [ ! -f ${cast_20_raw}/2020_manual_evaluation_topics_v1.0.json ]; then
        wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json
    fi
    if [ ! -f ${cast_20_raw}/2020qrels.txt ]; then
        wget https://trec.nist.gov/data/cast/2020qrels.txt
    fi
fi

cd ${base}
# python data/preprocess_cast19.py  --car_cbor=tmp2/datasets/raw/paragraphCorpus/dedup.articles-paragraphs.cbor  --msmarco_collection=tmp2/datasets/raw/msmarco.tsv  --duplicate_file=tmp2/datasets/raw/duplicate_list_v1.0.txt  --cast_dir=tmp2/datasets/raw/cast-19/  --out_data_dir=tmp2/datasets/cast-19  --out_collection_dir=tmp2/datasets/cast-shared
# python data/preprocess_cast20.py  --car_cbor=tmp2/datasets/raw/paragraphCorpus/dedup.articles-paragraphs.cbor  --msmarco_collection=tmp2/datasets/raw/msmarco.tsv  --duplicate_file=tmp2/datasets/raw/duplicate_list_v1.0.txt  --cast_dir=tmp2/datasets/raw/cast-20/  --out_data_dir=tmp2/datasets/cast-20  --out_collection_dir=tmp2/datasets/cast-shared

if [ ! -f ${or_quac_raw}/all_blocks.txt ] || [ ! -f ${or_quac_raw}/qrels.txt ]; then 
    cd ${or_quac_raw}
    if [ ! -f ${or_quac_raw}/all_blocks.txt ]; then
        wget https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz
    fi
    if [ ! -f ${or_quac_raw}/qrels.txt ]; then
        wget https://ciir.cs.umass.edu/downloads/ORConvQA/qrels.txt.gz
    fi
    gzip -d *.txt.gz
    mkdir preprocessed
    cd preprocessed
    wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt
    wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt
    wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt
fi

cd ${base}
# python data/preprocess_orquac.py  --orquac_dir=tmp2/datasets/raw/or-quac  --output_dir=tmp2/datasets/or-quac

# CAsT
# python data/tokenizing.py  --collection=tmp2/datasets/cast-shared/collection.tsv  --out_data_dir=tmp2/datasets/cast-shared/tokenized  --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-msmarco --model_type=rdot_nll
# OR-QuAC
# python data/tokenizing.py  --collection=tmp2/datasets/or-quac/collection.jsonl  --out_data_dir=tmp2/datasets/or-quac/tokenized  --model_name_or_path=bert-base-uncased --model_type=dpr

# CAsT
python -m torch.distributed.launch --nproc_per_node=$gpu_no drivers/gen_passage_embeddings.py  --data_dir=tmp2/datasets/cast-shared/tokenized  --checkpoint=tmp2/checkpoints/ad-hoc-ance-msmarco  --output_dir=tmp2/datasets/cast-shared/embeddings  --model_type=rdot_nll  --cache_dir=tmp2/datasets/cast-shared/cache
# OR-QuAC
python -m torch.distributed.launch --nproc_per_node=$gpu_no drivers/gen_passage_embeddings.py  --data_dir=tmp2/datasets/or-quac/tokenized  --checkpoint=tmp2/checkpoints/ad-hoc-ance-orquac.cp  --output_dir=tmp2/datasets/or-quac/embeddings  --model_type=dpr --cache_dir=tmp2/datasets/or-quac/cache
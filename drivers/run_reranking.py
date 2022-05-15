import torch
import json
import OpenMatch as om
from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer





def main(args):
    train_data = []
    with open(args.train_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            train_data.append(data)

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    input_ids = tokenizer.encode(train_data[0]["query"], train_data[0]["doc"])
    model = om.models.Bert("allenai/scibert_scivocab_uncased")
    ranking_score, ranking_features = model(torch.tensor(input_ids).unsqueeze(0))

    




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        help="path the the training file"
    )

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
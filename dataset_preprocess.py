"""
The script takes the conversation dataset and splits it into sentences.
It discards any sentences that:
- are not in the English language
- are longer than 48 tokens
- are shorter than 5 tokens

Note:
- The sentence splitting actually doesnt make too much sense
- The token bounds are suspicious
"""
import json
import nltk
from transformers import LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the text dataset")
    parser.add_argument("--model_name", type=str,default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--dataset_name", type=str, default="lmsys/lmsys-chat-1m")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process, 0 to process all")
    return parser.parse_args()

# TODO: This sentence splitting actually doesnt make too much sense
# check the first 10 samples and notice that it splits weirdly
# but we leave it here for now to reproduce the paper results
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# TODO: Are the max_length and min_length parameters important?
# Maybe the method doesnt work outside these bounds?
def process_data(text, tokenizer, max_length: int = 48, min_length: int = 5):
    sentences = split_into_sentences(text)
    processed_data = []
    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)
        if len(token_ids) > max_length or len(token_ids) < min_length:
            continue

        # TODO: If only one field is saved, whats the point of the dict?
        processed_data.append({
            "texts": sentence
        })
    return processed_data

def main(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    ds = load_dataset(dataset_name)

    for split in ["train"]:
        if args.num_samples > 0:
            ds_split = ds[split].select(range(args.num_samples))
        else:
            ds_split = ds[split]

        processed_data = []
        for example in tqdm(ds_split):
            if example['language'] != 'English':
                continue
            conversation = example["conversation"]
            for data in conversation:
                if data['role'] == 'user':
                    text = data['content']
                    processed_examples = process_data(text, tokenizer)
                    processed_data.extend(processed_examples)
            
        output_file = f"{args.output_dir}/{dataset_name.replace('/','__')}_{split}.json"
        with open(output_file, "w") as f:
            json.dump(processed_data, f)

        print(f"Dataset saved as: {output_file}")

if __name__ == "__main__":
    args = parse_args()
    nltk.download('punkt')
    main(args)

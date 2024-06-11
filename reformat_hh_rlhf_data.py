import glob
import gzip
import json
import argparse
import os
import random

import numpy as np
from itertools import product
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig


def dump_one_per_line(fp, data):
    fp.write(
        '[' +
        ',\n'.join(json.dumps(i) for i in data) +
        ']\n')


def get_query_response(tokenizer, hh_rlhf_sample, dataset_origin, max_query_length=64):
    processed_sample = dict()

    for i, (label, dialogue) in enumerate(hh_rlhf_sample.items()):
        split_on_assistant = dialogue.split("\n\nAssistant:")

        query = "\n\nAssistant:".join(split_on_assistant[:-1])
        response = "\n\nAssistant:" + split_on_assistant[-1]

        # TODO: Tiny-gpt2 max sequence length is 1024 truncate longer sequences
        query_tokens = tokenizer.encode(query, truncation=True)
        response_tokens = tokenizer.encode(response, truncation=True)

        # If the dialogue is too long to fit within the context of model consider
        if len(query_tokens) > max_query_length:
            reversed_query = query_tokens[::-1][:max_query_length]
            human_token = query_tokens[2] # \n\nHuman

            try:
                idx_human_token = reversed_query.index(human_token)
            except ValueError:
                continue

            while idx_human_token < max_query_length - 2:
                last_idx_human_token = idx_human_token
                reversed_query[idx_human_token] = 0

                try:
                    idx_human_token = reversed_query.index(human_token)
                except ValueError:
                    break

            query_tokens = query_tokens[:last_idx_human_token + 3]


        response_tokens = response_tokens[:1024 - max_query_length]

        query = tokenizer.pad(
            {"input_ids": query_tokens},
            padding="max_length",
            max_length=max_query_length,
            return_tensors=None,
            return_attention_mask=False,
        )
        response = tokenizer.pad(
            {"input_ids": response_tokens},
            padding="max_length",
            max_length=1024 - max_query_length,
            return_tensors=None,
            return_attention_mask=False,
        )

        if label == 'chosen':
            processed_sample['best'] = i

        processed_sample[f'sample{i}'] = response['input_ids']
        processed_sample['query'] = query['input_ids']
        assert len(processed_sample['query']) == max_query_length
        assert len(processed_sample[f'sample{i}']) == 1024 - max_query_length
        processed_sample['origin'] = dataset_origin
    if len(processed_sample) < 1:
        return None

    return processed_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hh_rlhf_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='sshleifer/tiny-gpt2')
    parser.add_argument('--max_query_len', type=int, default=745)

    args = parser.parse_args()
    max_query_length = args.max_query_len
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    gz_files = glob.glob(os.path.join(args.hh_rlhf_dir, '**/*.gz'), recursive=True)

    combined_data = list()
    for gz_file in gz_files:
        dataset_identifier = gz_file.split('.')[0].split('/')
        dataset_origin = dataset_identifier[-2]

        if 'base' not in dataset_origin:
            print(f"Skipped {gz_file}")
            continue

        print(f"Processing {gz_file}")
        with gzip.open(gz_file, mode="rt") as f:
            data = [
                get_query_response(tokenizer, json.loads(line), dataset_origin, max_query_length)
                for line in f
            ]
            data = list(filter(lambda x: x is not None, data))

            filename = "-".join(dataset_identifier[-2:])
            out_path = os.path.join(args.out_dir, f"{filename}.json")

            with open(out_path, 'w') as f:
                dump_one_per_line(f, data)
                # json.dump(data, f)
            combined_data.extend(data)

    out_path = os.path.join(args.out_dir, f"hh-base-combined.json")

    with open(out_path, 'w') as f:
        dump_one_per_line(f, combined_data)

    human_prefix = tokenizer.encode("\n\nHuman: Can you complete the following sentence with a positive sentiment?\n")
    assistant_prefix = tokenizer.encode("\n\nAssistant: Sure, here it is:\n")

    sentiment_data_path = os.path.join(args.hh_rlhf_dir, 'sentiment_offline_5k.json')

    with open(sentiment_data_path) as f:
        sentiment_data = json.load(f)

        processed_data = list()
        for sample in sentiment_data:
            processed_sample = dict()
            for feature, tokens in sample.items():
                if feature == 'best':
                    continue

                if 'sample' in feature:
                    # Responses
                    prefixed_tokens = assistant_prefix + tokens
                    length = 1024 - max_query_length
                else:
                    # Query
                    prefixed_tokens = human_prefix + tokens
                    length = max_query_length

                padded_feature =  tokenizer.pad(
                    {"input_ids": prefixed_tokens},
                    padding="max_length",
                    max_length=length,
                    return_tensors=None,
                    return_attention_mask=False,
                )

                processed_sample[feature] = padded_feature['input_ids']
                # processed_data.append(processed_sample)
            # processed_sample['origin'] = 'sentiment'
            best_response = [
                processed_sample[f"sample{sample['best']}"]
            ]
            other_responses = [
                processed_sample[f"sample{i}"] for i in range(4) if i != sample['best']
            ]

            all_pairs = [
                {
                    f'sample{idx}': best,
                    f'sample{1 - idx}': other,
                    'best': idx,
                    'query': processed_sample['query'],
                    'origin': 'sentiment'
                }
                for (best, other), idx in
                zip(product(best_response, other_responses), np.random.randint(2, size=(3,), dtype=np.int32).tolist())
            ]

            processed_data.extend(all_pairs)

        random.shuffle(processed_data)

        len_sentiment_data = len(processed_data)
        num_train = round(len_sentiment_data * 0.8)
        num_test = len_sentiment_data - num_train
        processed_test_data = processed_data[num_train:]
        processed_data = processed_data[:num_train]

        print(len(processed_data), len(processed_test_data), num_train, num_test)
        sentiment_data_filename = os.path.basename(sentiment_data_path)
        out_path = os.path.join(args.out_dir, sentiment_data_filename)
        with open(out_path, 'w') as f:
            dump_one_per_line(f, processed_data)
            # json.dump(processed_data, f)

        out_path = os.path.join(args.out_dir, 'test.json')
        with open(out_path, 'w') as f:
            dump_one_per_line(f, processed_test_data)

        combined_data.extend(processed_data)

    out_path = os.path.join(args.out_dir, "sentiment-base-combined.json")
    with open(out_path, 'w') as f:
        dump_one_per_line(f, combined_data)
        # json.dump(combined_data, f)

import json
import numpy as np
import random
import argparse
import torch
from collections import defaultdict
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from preprocess_utils import DataRow


def create_adjacency_list(train_data):
    """
    Use QA train json file to create an adjacency list of the form:
    {source_node: [DataRow1, DataRow2, ...]}
    """
    adj_list = defaultdict(list)
    for data_row in train_data:
        data_row = DataRow(**data_row)
        adj_list[data_row.source].append(data_row)

    return adj_list, len(train_data)


def get_similar_pairs_linked(adjacency_list, train_set_count):
    """
    Use adjacency list to generate similar pairs of linked nodes:
    {
        [[{ Info_Node1 }, { Info_Node2 }]]
    }
    The target of node1 is the source of node2
    """
    pair_limit = int(train_set_count / 2)
    similar_pairs = []

    next_index = {}
    for source in adjacency_list.keys():
        next_index[source] = 0
    
    set_of_links = set()
    num_pairs_chosen = 0
    pbar = tqdm(total=pair_limit)
    while (num_pairs_chosen < pair_limit):
        for source, info_list in adjacency_list.items():
            if next_index[source] >= len(info_list):
                next_index[source] = 0

            info = info_list[next_index[source]]
            next_index[source] += 1

            if info.target in adjacency_list and info.source != info.target and info.answer == 'yes':
                index = random.randint(0, len(adjacency_list[info.target]) - 1)
                match = adjacency_list[info.target][index]
                link = (source, info.target, match.target)
                if link not in set_of_links:
                    similar_pairs.append((info, match))
                    set_of_links.add(link)
                    num_pairs_chosen += 1
                    pbar.update(1)
                    break  
    pbar.close()
    
    print("Total Num Pairs: ", len(similar_pairs))
    return similar_pairs


def get_similar_pairs_adjacent(adjacency_list):
    """
    Use adjacency list to generate similar pairs of node edges:
    {
        [[{ Info_Node1 }, { Info_Node2 }]]
    }
    The source of node1 is equal to the source of node2
    """
    similar_pairs = []
    max_num_pairs = float("-inf")
    pbar = tqdm(total=len(adjacency_list.items()))
    for source, info_list in adjacency_list.items():
        num_pairs_count = 0
        for i in range(0, len(info_list), 2):
            if i + 1 < len(info_list):
                similar_pairs.append((info_list[i], info_list[i + 1]))
            else:
                similar_pairs.append((info_list[0], info_list[i]))
            num_pairs_count += 1
        max_num_pairs = max(max_num_pairs, num_pairs_count)
        pbar.update(1)
    pbar.close()

    print("Total Num Pairs:", len(similar_pairs))
    print("Max Pairs Per Source", max_num_pairs)
    return similar_pairs


def linked_similarity_cosine_stats(linked_similarity_data):
    """
    Get cosine similarity score statistics
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    questions = []
    cosine_values = []
    for pair in linked_similarity_data:
        questions = [pair[0]["question"], pair[1]["question"]]
        inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        cosine_sim = 1 - cosine(embeddings[0], embeddings[1])
        cosine_values.append(cosine_sim)

    print("Cosine Min: ", np.min(cosine_values))
    print("Cosine Max: ", np.max(cosine_values))
    print("Cosine Mean: ", np.mean(cosine_values))
    print("Cosine Median: ", np.median(cosine_values))
    print("Cosine Stddev: ", np.std(cosine_values))
    print("Cosine Variance: ", np.var(cosine_values))


def get_similar_pairs_cosine(train_data, window_size=8000):
    """
    Use SimCSE to get cosine similarity of question embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    questions = [data['question'] for data in train_data]

    # Get the embeddings
    embeddings = []
    start = 0
    end = start + window_size
    with torch.no_grad():
        while start < len(questions):
            question_chunk = questions[start:end]
            inputs = tokenizer(question_chunk, padding=True, truncation=True, return_tensors="pt")

            print("Retrieve Embeddings ", start)
            embed = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

            embeddings.append(embed)
            start = end
            end += window_size

    threshold = 0.652
    similar_pairs_index = set()
    
    num_questions = len(questions)
    pair_limit = int(num_questions / 2)
    print("Retrieving Pairs")
    pbar = tqdm(total=pair_limit)
    while len(similar_pairs_index) < pair_limit:
        questionsIndex = random.sample(range(len(questions)), 2)
        cosine_sim = 1 - cosine(embeddings[questionsIndex[0]], embeddings[questionsIndex[1]])
        if cosine_sim >= threshold:
            similar_pairs_index.add(tuple(questionsIndex))
            pbar.update(1)
    pbar.close()
    
    similar_pairs = []
    for pair in list(similar_pairs_index):
        similar_pairs.append([train_data[pair[0]], train_data[pair[1]]])

    return similar_pairs


def json_serialize_pairs(question_pairs):
    return [(q1._asdict(), q2._asdict()) for (q1, q2) in question_pairs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="./beliefbank-data-sep2021/qa_train.json")
    parser.add_argument('--method', required=True, choices=["linked", "adjacent", "cosine", "cosine_large"])
    parser.add_argument('--output_path', default="./beliefbank-data-sep2021/qa_train.json")
    args = parser.parse_args()

    # Beliefbank paths
    # beliefbank-data-sep2021/qa_train_sim_linked.json
    # beliefbank-data-sep2021/qa_train_sim_adjacent.json
    # beliefbank-data-sep2021/qa_train_similar_sim_cosine.json

    # LoT paths
    # leap_of_thought_data/train_sim_linked.json
    # leap_of_thought_data/train_sim_adjacent.json
    # leap_of_thought_data/train_sim_simcse.json

    with open(args.train_path) as f:
        train_data = json.load(f)

    if args.method == "linked":
        adjacency_list, train_set_count = create_adjacency_list(train_data)

        similar_pairs_linked = get_similar_pairs_linked(adjacency_list, train_set_count)

        with open(args.output_path, 'w') as f:
            json.dump(json_serialize_pairs(similar_pairs_linked), f, indent=1)

    elif args.method == "adjacent":
        adjacency_list, train_set_count = create_adjacency_list(train_data)

        similar_pairs_adj = get_similar_pairs_adjacent(adjacency_list)

        with open(args.output_path, 'w') as f:
            json.dump(json_serialize_pairs(similar_pairs_adj), f, indent=1)
    
    elif args.method == "cosine_stats":
        linked_similarity_cosine_stats(train_data)
    
    elif args.method == "cosine":
        similar_pairs_cosine = get_similar_pairs_cosine(train_data)

        with open(args.output_path, 'w') as f:
            json.dump(similar_pairs_cosine, f, indent=1)



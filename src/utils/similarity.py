import json
import numpy as np
import random
import argparse
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


def create_adjacency_list(train_data):
    """
    Use QA train json file to create an adjacency list of the form:
    {
        source_node: [
            {
                "question": question,
                "answer": answer,
                "source": source_node,
                "target": target_node,
                "gold": gold
            }, ...
        ]
    }
    """
    adjacency_list = {}
    count = 0
    for data in train_data:
        if data["source"] not in adjacency_list:
            adjacency_list[data["source"]] = []

        info_dict = {
            "question": data["question"],
            "answer": data["answer"],
            "source": data["source"],
            "target": data["target"],
            "gold": data["gold"]
        }

        adjacency_list[data["source"]].append(info_dict)
        count += 1
    print("Original Train Num:", count)

    return adjacency_list, count


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
    while (num_pairs_chosen < pair_limit):
        for source, info_list in adjacency_list.items():
            if next_index[source] >= len(info_list):
                next_index[source] = 0

            info = info_list[next_index[source]]
            next_index[source] += 1

            if info["target"] in adjacency_list.keys() and info["source"] != info["target"] and info["answer"] == 'yes':
                index = random.randint(0, len(adjacency_list[info["target"]]) - 1)
                match = adjacency_list[info["target"]][index]
                link = (source, info["target"], match["target"])
                if link not in set_of_links:
                    similar_pairs.append((info, match))
                    set_of_links.add(link)
                    num_pairs_chosen += 1
                    break  
    
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
    for source, info_list in adjacency_list.items():
        num_pairs_count = 0
        for i in range(0, len(info_list), 2):
            if i + 1 < len(info_list):
                similar_pairs.append((info_list[i], info_list[i + 1]))
            else:
                similar_pairs.append((info_list[0], info_list[i]))
            num_pairs_count += 1
        max_num_pairs = max(max_num_pairs, num_pairs_count)

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


def get_similar_pairs_cosine(train_data):
    """
    Use SimCSE to get cosine similarity of question embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    questions = []
    for data in train_data:
        questions.append(data["question"])
    
    inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    print("Retrieve Embeddings")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    threshold = 0.7
    similar_pairs_index = set()
    
    num_questions = len(questions)
    pair_limit = int(num_questions / 2)
    pair_limit = 1
    print("Retrieving Pairs")
    while len(similar_pairs_index) < pair_limit:
        questionsIndex = random.sample(range(len(questions)), 2)
        print(questionsIndex)
        cosine_sim = 1 - cosine(embeddings[questionsIndex[0]], embeddings[questionsIndex[1]])
        if cosine_sim >= threshold:
            similar_pairs_index.add(tuple(questionsIndex))
    
    similar_pairs = []
    for pair in list(similar_pairs_index):
        similar_pairs.append([train_data[pair[0]], train_data[pair[1]]])

    return similar_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="./beliefbank-data-sep2021/qa_train.json")
    parser.add_argument('--method', default="") # Options: adjacent, linked, cosine
    args = parser.parse_args()

    with open(args.train_path) as f:
        train_data = json.load(f)

    if (args.method == "linked"):
        adjacency_list, train_set_count = create_adjacency_list(train_data)

        similar_pairs_linked = get_similar_pairs_linked(adjacency_list, train_set_count)

        with open('beliefbank-data-sep2021/qa_train_similar_linked.json', 'w') as f:
            json.dump(similar_pairs_linked, f, indent=1)

    elif (args.method == "adjacent"):
        adjacency_list, train_set_count = create_adjacency_list(train_data)

        similar_pairs_adj = get_similar_pairs_adjacent(adjacency_list)

        with open('beliefbank-data-sep2021/qa_train_similar_adjacent.json', 'w') as f:
            json.dump(similar_pairs_adj, f, indent=1)
    
    elif (args.method == "cosine_stats"):
        linked_similarity_cosine_stats(train_data)
    
    elif (args.method == "cosine"):
        similar_pairs_cosine = get_similar_pairs_cosine(train_data)

        with open('beliefbank-data-sep2021/qa_train_similar_cosine.json', 'w') as f:
            json.dump(similar_pairs_cosine, f, indent=1)
    
    


    

import json
from re import A


def get_similar_pairs_linked(train_data):
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
    print(count)
    
    similar_pairs = []
    ignored_count = 0
    max_num_pairs = float("-inf")
    for source, info_list in adjacency_list.items():
        for info in info_list:
            num_pairs_count = 0
            if info["target"] in adjacency_list and info["answer"] == 'yes':
                for match in adjacency_list[info["target"]]:
                    similar_pairs.append((info, match))
                    num_pairs_count += 1
            else:
                ignored_count += 1
            max_num_pairs = max(max_num_pairs, num_pairs_count)

    print("Ignored Count:", str(ignored_count))
    print("Total Pairs:", len(similar_pairs))
    print("Max Pairs", max_num_pairs)
    return similar_pairs


if __name__ == "__main__":
    with open("./beliefbank-data-sep2021/qa_train.json") as f:
        train_data = json.load(f)

    similar_pairs = get_similar_pairs_linked(train_data)

    with open('beliefbank-data-sep2021/qa_train_similar_linked.json', 'w') as f:
        json.dump(similar_pairs, f, indent=1)

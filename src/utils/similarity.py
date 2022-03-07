import json
import random


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

    # for source, info_list in adjacency_list.items():
        # random.shuffle(info_list)

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


if __name__ == "__main__":
    with open("./beliefbank-data-sep2021/qa_train.json") as f:
        train_data = json.load(f)

    adjacency_list, train_set_count = create_adjacency_list(train_data)

    similar_pairs_linked = get_similar_pairs_linked(adjacency_list, train_set_count)

    with open('beliefbank-data-sep2021/qa_train_similar_linked.json', 'w') as f:
        json.dump(similar_pairs_linked, f, indent=1)

    # similar_pairs_adj = get_similar_pairs_adjacent(adjacency_list)

    # with open('beliefbank-data-sep2021/qa_train_similar_adjacent.json', 'w') as f:
    #     json.dump(similar_pairs_adj, f, indent=1)

    

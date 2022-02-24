import json


def check_constraint_link_pairs():
    """
    Check the possible link pairs in constraints (i.e. (IsA, HasA, forward))
    """
    with open("../beliefbank-data-sep2021/constraints_v2.json") as f:
        constraints = json.load(f)

    constraint_links = constraints['links']
    link_pairs = set()
    for link_dict in constraint_links:
        source = link_dict["source"].split(',')[0]
        target = link_dict["target"].split(',')[0]
        direction = link_dict["direction"]
        link_pairs.add((source, target, direction))

    print(link_pairs)


def questions_to_source_target_node():
    """
    Create dictionary of questions to source and target nodes
    """
    with open("../beliefbank-data-sep2021/qa.json") as f:
        qa_info = json.load(f)
    
    question_to_node_dict = {}
    for qa in qa_info:
        question = qa['question'][:-2] + '?'
        question_to_node_dict[question] = {
            'source': qa['source'],
            'target': qa['target']
        }

    with open('../beliefbank-data-sep2021/qa_to_nodes.json', 'w') as f:
        json.dump(question_to_node_dict, f)

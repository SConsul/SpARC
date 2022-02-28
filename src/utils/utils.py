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


def questions_to_source_target_node(qa_file, question_to_node_dict):
    """
    Create dictionary of questions to source and target nodes
    """
    with open(qa_file) as f:
        qa_info = json.load(f)
    
    for qa in qa_info:
        question = qa['question']
        question_to_node_dict[question] = {
            'source': qa['source'],
            'target': qa['target']
        }
    
    return question_to_node_dict
    

def get_all_qa_nodes():
    question_to_node_dict = {}
    question_to_node_dict = questions_to_source_target_node("../beliefbank-data-sep2021/qa_train.json", question_to_node_dict)

    question_to_node_dict = questions_to_source_target_node("../beliefbank-data-sep2021/qa_val.json", question_to_node_dict)

    question_to_node_dict = questions_to_source_target_node("../beliefbank-data-sep2021/qa_test.json", question_to_node_dict)

    with open('../beliefbank-data-sep2021/qa_to_nodes.json', 'w') as f:
        json.dump(question_to_node_dict, f)


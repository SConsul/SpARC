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
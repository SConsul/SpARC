import json
import random
import numpy as np
from itertools import product
from collections import defaultdict, namedtuple


Edge = namedtuple('DataRow', ['question', 'answer', 'source', 'target', 'gold'])


class DataRow(Edge):
    def __hash__(self):
        # return hash(self.question)
        return hash((self.source, self.target))

    def __eq__(self, other):
        # return self.question == other.question
        return self.source == other.source and self.target == other.target


TEMPLATES = {
    "IsA": {"assertion_positive": "[X] is [Y].", "assertion_negative":  "[X] is not [Y].",
            "templates": ["Is [X] [Y]?", "Is it true that [X] is [Y]?"],
            "templates_negated": ["Is [X] not [Y]?", "Is it true that [X] is not [Y]?"]},
    "HasPart": {"assertion_positive": "[X] has [Y].", "assertion_negative":  "[X] does not have [Y].",
                "templates": ["Does [X] have [Y]?", "Is [Y] part of [X]?",
                              "Is it true that [Y] is part of [X]?", "Is it true that [X] has [Y]?"],
                "templates_negated": ["Does [X] not have [Y]?", "Is [Y] not part of [X]?",
                                      "Is it true that [Y] is not part of [X]?",
                                      "Is it true that [X] does not have [Y]?"]},
    "CapableOf": {"assertion_positive": "[X] can [Y].", "assertion_negative":  "[X] cannot [Y].",
                  "templates": ["Can [X] [Y]?", "Is it true that [X] can [Y]?",
                                "Is it true that [X] is capable of [Y]?", "Is [X] capable of [Y]?",
                                "Is [X] able to [Y]?", "Is it true that [X] is able to [Y]?"],
                  "templates_negated": ["Can [X] not [Y]?", "Is it true that [X] cannot [Y]?",
                                        "Is it true that [X] is not capable of [Y]?",
                                        "Is [X] not capable of [Y]?",
                                        "Is [X] able not to [Y]?",
                                        "Is it true that [X] is not able to [Y]?"]},
    "MadeOf": {"assertion_positive": "[X] is made of [Y].", "assertion_negative":  "[X] is not made of [Y].",
               "templates": ["Is [X] made of [Y]?", "Is it true that [X] is made of [Y]?"],
               "templates_negated": ["Is [X] not made of [Y]?", "Is it true that [X] is not made of [Y]?"]},
    "HasProperty": {"assertion_positive": "[X] is [Y].", "assertion_negative":  "[X] is not [Y].",
                    "templates": ["Is [X] [Y]?", "Is it true that [X] is [Y]?"],
                    "templates_negated": ["Is [X] not [Y]?", "Is it true that [X] is not [Y]?"]},
    "HasA": {"assertion_positive": "[X] has [Y].", "assertion_negative":  "[X] does not have [Y].",
             "templates": ["Does [X] have [Y]?", "Is it true that [X] has [Y]?"],
             "templates_negated": ["Does [X] not have [Y]?", "Is it true that [X] has not [Y]?"]}
}


def parse_source_target(source, target, use_pos=True):
    # ASSUME: source is always "IsA"
    # Use non_countable
    _, s_obj = source.split(',')
    link, t_obj = target.split(',')

    template_qs = TEMPLATES[link]['templates' if use_pos else 'templates_negated']
    question = random.choice(template_qs)

    s_art = 'an ' if s_obj[0] in {'a', 'e', 'i', 'o', 'u'} else 'a '
    s_art = '' if s_obj in non_countable else s_art
    t_art = 'an ' if t_obj[0] in {'a', 'e', 'i', 'o', 'u'} else 'a '
    t_art = '' if t_obj in non_countable else t_art

    return question.replace('[X]', s_art + s_obj).replace('[Y]', t_art + t_obj)


def traverse(data_adj_list):
    visited = defaultdict(set)

    def _traverse_dfs(source, n):
        for e in data_adj_list.get(n):
            s_n_edge = DataRow(question=parse_source_target(source, e.target), answer=e.answer,
                               source=source, target=e.target, gold=e.source == source)

            if s_n_edge not in visited[source]:
                visited[source].add(s_n_edge)

                if s_n_edge.target in data_adj_list and s_n_edge.answer == 'yes':
                    _traverse_dfs(source, s_n_edge.target)
        return

    for n in data_adj_list:
        _traverse_dfs(n, n)

    return visited


def process_c_graph(c_graph):
    data = defaultdict(set)
    nodes = c_graph['nodes']
    edges = c_graph['links']

    for e in edges:
        s, t = (e['source'], e['target']) if e['direction'] == 'forward' else (e['target'], e['source'])
        impl = e['weight']

        row = DataRow(question=parse_source_target(s, t), answer='yes' if impl == 'yes_yes' else 'no',
                      source=s, target=t, gold=True)
        # row = {'question': parse_source_target(s, t), 'answer': 'yes' if impl == 'yes_yes' else 'no',
               # 'source': s, 'target': t}
        data[s].add(row)

    # Add silver edges
    return data, traverse(data)


def create_all_questions(c_graph):
    nodes = [n['id'] for n in c_graph['nodes']]
    isa_nodes = [n for n in nodes if n.startswith('IsA')]
    all_pairs = product(isa_nodes, nodes)

    all_questions = defaultdict(set)
    for source, target in all_pairs:
        all_questions[source].add(DataRow(question=parse_source_target(source, target), answer='n/a',
                                          source=source, target=target, gold=False))
    return all_questions


def process_silver_facts(silver_facts):
    # Correct imbalance in silver facts by ensuring equal number of yes qs and no qs
    num_yes, num_no = 0, 0
    use_positive_question = {}
    for source, targets in silver_facts.items():
        for target, label in targets.items():
            use_positive_question['IsA,' + source + target] = label
            num_yes += 1 if label == 'yes' else 0
            num_no += 1 if label == 'no' else 0

    keys = list(use_positive_question.keys())
    random.shuffle(keys)

    imbalance_ans = 'yes' if num_yes >= num_no else 'no'
    flip_ans = lambda a: 'yes' if a == 'no' else 'no'

    i = 0
    count_flipped = 0
    while count_flipped < (max(num_yes, num_no) - min(num_yes, num_no))/2:
        k = keys[i]
        ans = use_positive_question[k]
        if ans == imbalance_ans:
            use_positive_question[k] = False
            count_flipped += 1
        else:
            use_positive_question[k] = True
        i += 1

    data = defaultdict(set)
    for source, targets in silver_facts.items():
        for target, label in targets.items():
            # If use_pos, then use template, else use negative template and flip answer
            use_pos = use_positive_question['IsA,' + source + target]
            t_type, t_obj = target.split(',')
            target_ = t_type + ',' + ('' if use_pos else 'not ') + t_obj
            row = DataRow(question=parse_source_target('IsA,' + source, target, use_pos=use_pos),
                          answer=label if use_pos else flip_ans(label),
                          source='IsA,' + source, target=target_, gold=False)
            data['IsA,' + source].add(row)

    # data = {n: [q._asdict() for q in qs] for n, qs in data.items()}
    return data


def graph_split(all_adj_list):
    val_adj_list = defaultdict(set)
    test_adj_list = defaultdict(set)

    available_nodes, available_edges = set(), set()
    for n, edges in all_adj_list.items():
        available_edges.update(edges)
        for data_row in edges:
            available_nodes.add(data_row.source)
            available_nodes.add(data_row.target)
    print(f"Total nodes: {len(available_nodes)}, total edges: {len(available_edges)}")
    n_nodes = len(available_nodes)

    start_nodes = np.random.choice(list(all_adj_list.keys()), size=2, replace=False)
    for s in start_nodes:
        available_nodes.remove(s)

    val_degree_nodes = {start_nodes[0]}
    val_nodes = {start_nodes[0]}

    test_degree_nodes = {start_nodes[1]}
    test_nodes = {start_nodes[1]}

    def depth_step(source, split_adj_list, split_degree_nodes, split_nodes, other_nodes):
        if source is None:
            return

        neighbours = all_adj_list.get(source, [])

        for data_row in neighbours:
            target = data_row.target

            # Add edge if target is not in other split and if edge is available
            if target not in other_nodes and data_row in available_edges:
                split_adj_list[source].add(data_row)
                available_edges.remove(data_row)

                split_nodes.add(target)
                split_degree_nodes.add(target)
                available_nodes.discard(target)

                return target

        # Discard source as restart option because no outgoing edges left to explore here
        split_degree_nodes.discard(source)

        # Prefer nodes along graph already in split before completely random restart
        if len(split_degree_nodes) > 0:
            restart = random.choice(list(split_degree_nodes))
            split_nodes.add(restart)
            split_degree_nodes.add(restart)
            available_nodes.discard(restart)
            return restart
        if len(available_nodes) > 0:
            restart = random.choice(list(available_nodes))
            split_nodes.add(restart)
            split_degree_nodes.add(restart)
            available_nodes.remove(restart)
            return restart

        print("No more nodes")
        return None

    curr_val = start_nodes[0]
    curr_test = start_nodes[0]
    while len(val_nodes) + len(test_nodes) < n_nodes:
        curr_val = depth_step(curr_val, val_adj_list, val_degree_nodes, val_nodes, test_nodes)
        curr_test = depth_step(curr_test, test_adj_list, test_degree_nodes, test_nodes, val_nodes)

    conflict = val_nodes.intersection(test_nodes)
    assert len(conflict) == 0, f"Conflict: {len(conflict)}, Intersection: {conflict}"

    n_val_edges = sum([len(edges) for edges in val_adj_list.values()])
    n_test_edges = sum([len(edges) for edges in test_adj_list.values()])

    print(f"Missing edges: {len(available_edges)}, val edges: {n_val_edges}, test edges: {n_test_edges}")
    print(f"Val nodes: {len(val_nodes)}, test nodes: {len(test_nodes)}")
    return val_adj_list, test_adj_list


def data_split(data, train=0.8, val=0.1, test=0.1):
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)

    for n, questions in data.items():
        q_prime = list(questions.copy())
        random.shuffle(q_prime)
        tr_a, tr_b = 0, int(train * len(q_prime))
        val_a, val_b = tr_b, tr_b + int(val * len(q_prime))
        test_a, test_b = val_b, len(q_prime)

        train_data[n].extend(q_prime[tr_a:tr_b])
        val_data[n].extend(q_prime[val_a:val_b])
        test_data[n].extend(q_prime[test_a:test_b])

    return train_data, val_data, test_data


def merge(data1, data2):
    new_data = defaultdict(set)
    for k, qs in data1.items():
        new_data[k].update(qs)

    for k, qs in data2.items():
        new_data[k].update(qs)

    return new_data


def json_serialize(data):
    return {n: [q._asdict() for q in qs] for n, qs in data.items()}


def flatten(l):
    return [x for sub_l in l for x in sub_l]


if __name__ == "__main__":
    with open('beliefbank-data-sep2021/constraints_v2.json', 'r') as f:
        c_graph = json.load(f)

    with open('beliefbank-data-sep2021/silver_facts.json', 'r') as f:
        facts = json.load(f)

    with open('beliefbank-data-sep2021/non_countable.txt', 'r') as f:
        non_countable = f.readlines()

    c_adj_list, c_data = process_c_graph(c_graph)

    s_data = process_silver_facts(facts)

    c_multi_hop = defaultdict(set)
    for n, all_edges in c_data.items():
        c_multi_hop[n] = all_edges - c_adj_list[n]

    # Train data is most of silver facts (i.e. actual questions with entities)
    train, s_val, s_test = data_split(s_data, train=0.9, val=0.05, test=0.05)

    # Eval data is all edges in constraint graph (single and multi hop)
    # _, val, test = data_split(c_adj_list, train=0., val=0.5, test=0.5)
    val, test = graph_split(c_adj_list)

    # Consistency data is dense graph of all questions starting with isA
    # consistency_data = create_all_questions(c_graph)

    # with open('beliefbank-data-sep2021/silver_qa.json', 'w') as f:
    #     json.dump(json_serialize(s_data), f)
    #
    # # with open('beliefbank-data-sep2021/qa.json', 'w') as f:
    # #     json.dump(flatten(data.values()), f, indent=1)
    #
    # with open('beliefbank-data-sep2021/qa_train.json', 'w') as f:
    #     json.dump(flatten(json_serialize(train).values()), f, indent=1)
    #
    # with open('beliefbank-data-sep2021/constraints_qa.json', 'w') as f:
    #     json.dump(json_serialize(c_adj_list), f, indent=1)
    #
    # with open('beliefbank-data-sep2021/constraints_qa_multihop.json', 'w') as f:
    #     json.dump(json_serialize(c_data), f, indent=1)
    #
    # with open('beliefbank-data-sep2021/qa_val.json', 'w') as f:
    #     json.dump(flatten(json_serialize(val).values()), f, indent=1)
    #
    # with open('beliefbank-data-sep2021/qa_test.json', 'w') as f:
    #     json.dump(flatten(json_serialize(test).values()), f, indent=1)

    # with open('beliefbank-data-sep2021/qa_consistency.json', 'w') as f:
    #     json.dump(flatten(json_serialize(consistency_data).values()), f, indent=1)



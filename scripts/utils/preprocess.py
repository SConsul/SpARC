import json
import random
from collections import defaultdict, namedtuple


Edge = namedtuple('DataRow', ['question', 'answer', 'source', 'target', 'gold'])


class DataRow(Edge):
    def __hash__(self):
        return hash(self.question)

    def __eq__(self, other):
        return self.question == other.question


def parse_source_target(source, target):
    # ASSUME: source is always "IsA"
    s_obj, t_obj = source.split(',')[1], target.split(',')[1]

    s_art = 'an ' if s_obj[0] in {'a', 'e', 'i', 'o', 'u'} else 'a '
    s_art = s_art if s_obj[-1] != 's' else ''
    t_art = 'an ' if t_obj[0] in {'a', 'e', 'i', 'o', 'u'} else 'a '
    t_art = t_art if t_obj[-1] != 's' else ''
    if target.startswith('IsA'):
        return f'Is {s_art}{s_obj} {t_art}{t_obj}?'
    elif target.startswith('HasA'):
        return f'Does {s_art}{s_obj} have {t_art}{t_obj}?'
    elif target.startswith('HasPart'):
        return f'Does {s_art}{s_obj} have {t_art}{t_obj}?'
    elif target.startswith('HasProperty'):
        return f'Is {s_art}{s_obj} {t_obj}?'
    elif target.startswith('MadeOf'):
        return f'Is {s_art}{s_obj} made of {t_obj}?'
    elif target.startswith('CapableOf'):
        return f'Can {s_art}{s_obj} {t_obj}?'
    else:
        raise ValueError("NOOOOO")


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


def process_silver_facts(silver_facts):
    data = defaultdict(set)
    for source, targets in silver_facts.items():
        for target, label in targets.items():
            row = DataRow(question=parse_source_target('IsA,' + source, target), answer=label,
                          source='IsA,' + source, target=target, gold=False)
            data['IsA,' + source].add(row)

    # data = {n: [q._asdict() for q in qs] for n, qs in data.items()}
    return data


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

    c_adj_list, c_data = process_c_graph(c_graph)

    s_data = process_silver_facts(facts)

    c_multi_hop = defaultdict(set)
    for n, all_edges in c_data.items():
        c_multi_hop[n] = all_edges - c_adj_list[n]

    # Merge multihop with silver data
    eval_data = merge(c_multi_hop, s_data)

    # Split single hop edges into train/val/test
    train, one_hop_val, one_hop_test = data_split(c_adj_list)
    _, multi_val, multi_test = data_split(eval_data, train=0., val=0.5, test=0.5)
    val = merge(one_hop_val, multi_val)
    test = merge(one_hop_test, multi_test)
    
    train = json_serialize(train)
    val = json_serialize(val)
    test = json_serialize(test)

    with open('beliefbank-data-sep2021/constraints_qa.json', 'w') as f:
        json.dump(json_serialize(c_data), f)

    with open('beliefbank-data-sep2021/silver_qa.json', 'w') as f:
        json.dump(json_serialize(s_data), f)

    # with open('beliefbank-data-sep2021/qa.json', 'w') as f:
    #     json.dump(flatten(data.values()), f, indent=1)

    with open('beliefbank-data-sep2021/qa_train.json', 'w') as f:
        json.dump(flatten(train.values()), f, indent=1)

    with open('beliefbank-data-sep2021/qa_val.json', 'w') as f:
        json.dump(flatten(val.values()), f, indent=1)

    with open('beliefbank-data-sep2021/qa_test.json', 'w') as f:
        json.dump(flatten(test.values()), f, indent=1)



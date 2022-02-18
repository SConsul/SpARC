import json
import random
from collections import defaultdict, namedtuple


data_row = namedtuple('DataRow', ['question', 'answer', 'source', 'target'])


def parse_source_target(source, target):
    # ASSUME: source is always "IsA"
    s_obj, t_obj = source.split(',')[1], target.split(',')[1]

    s_art = 'an' if s_obj[0] in {'a', 'e', 'i', 'o', 'u'} else 'a'
    s_art = s_art if s_obj[-1] != 's' else ''
    t_art = 'an' if t_obj[0] in {'a', 'e', 'i', 'o', 'u'} else 'a'
    t_art = t_art if t_obj[-1] != 's' else ''
    if target.startswith('IsA'):
        return f'Is {s_art} {s_obj} {t_art} {t_obj} ?'
    elif target.startswith('HasA'):
        return f'Does {s_art} {s_obj} have {t_art} {t_obj} ?'
    elif target.startswith('HasPart'):
        return f'Does {s_art} {s_obj} have {t_art} {t_obj} ?'
    elif target.startswith('HasProperty'):
        return f'Is {s_art} {s_obj} {t_obj} ?'
    elif target.startswith('MadeOf'):
        return f'Is {s_art} {s_obj} made of {t_obj} ?'
    elif target.startswith('CapableOf'):
        return f'Can {s_art} {s_obj} {t_obj} ?'
    else:
        raise ValueError("NOOOOO")


def process_c_graph(c_graph):
    data = defaultdict(set)
    nodes = c_graph['nodes']
    edges = c_graph['links']

    for e in edges:
        s, t = (e['source'], e['target']) if e['direction'] == 'forward' else (e['target'], e['source'])
        impl = e['weight']

        row = data_row(question=parse_source_target(s, t), answer='yes' if impl == 'yes_yes' else 'no',
                       source=s, target=t)
        # row = {'question': parse_source_target(s, t), 'answer': 'yes' if impl == 'yes_yes' else 'no',
               # 'source': s, 'target': t}
        data[s].add(row)

    data = {n: [q._asdict() for q in qs] for n, qs in data.items()}
    return data


def process_silver_facts(silver_facts):
    data = defaultdict(set)
    for source, targets in silver_facts.items():
        for target, label in targets.items():
            row = data_row(question=parse_source_target('IsA,' + source, target), answer=label,
                           source='IsA,' + source, target=target)
            data['IsA,' + source].add(row)

    data = {n: [q._asdict() for q in qs] for n, qs in data.items()}
    return data


def data_split(data, train=0.8, val=0.1, test=0.1):
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)

    for n, questions in data.items():
        q_prime = questions.copy()
        random.shuffle(q_prime)
        tr_a, tr_b = 0, int(train * len(q_prime))
        val_a, val_b = tr_b, tr_b + int(val * len(q_prime))
        test_a, test_b = val_b, len(q_prime)

        train_data[n].extend(q_prime[tr_a:tr_b])
        val_data[n].extend(q_prime[val_a:val_b])
        test_data[n].extend(q_prime[test_a:test_b])

    return train_data, val_data, test_data


def flatten(l):
    return [x for sub_l in l for x in sub_l]


if __name__ == "__main__":
    with open('beliefbank-data-sep2021/constraints_v2.json', 'r') as f:
        c_graph = json.load(f)

    with open('beliefbank-data-sep2021/silver_facts.json', 'r') as f:
        facts = json.load(f)

    c_data = process_c_graph(c_graph)
    s_data = process_silver_facts(facts)

    train_c, val_c, test_c = data_split(c_data)
    train_s, val_s, test_s = data_split(s_data)

    with open('beliefbank-data-sep2021/constraints_qa.json', 'w') as f:
        json.dump(c_data, f)

    with open('beliefbank-data-sep2021/silver_qa.json', 'w') as f:
        json.dump(s_data, f)

    with open('beliefbank-data-sep2021/qa.json', 'w') as f:
        json.dump(flatten(c_data.values()) + flatten(s_data.values()), f, indent=1)

    with open('beliefbank-data-sep2021/qa_train.json', 'w') as f:
        json.dump(flatten(train_c.values()) + flatten(train_s.values()), f, indent=1)

    with open('beliefbank-data-sep2021/qa_val.json', 'w') as f:
        json.dump(flatten(val_c.values()) + flatten(val_s.values()), f, indent=1)

    with open('beliefbank-data-sep2021/qa_test.json', 'w') as f:
        json.dump(flatten(test_c.values()) + flatten(test_s.values()), f, indent=1)



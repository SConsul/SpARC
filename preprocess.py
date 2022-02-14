import json


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
    data = []
    nodes = c_graph['nodes']
    edges = c_graph['links']

    for e in edges:
        s, t = (e['source'], e['target']) if e['direction'] == 'forward' else (e['target'], e['source'])
        impl = e['weight']

        row = {'question': parse_source_target(s, t), 'answer': 'yes' if impl == 'yes_yes' else 'no'}
        data.append(row)

    return data


def process_silver_facts(silver_facts):
    data = []
    for source, targets in silver_facts.items():
        for target, label in targets.items():
            row = {'question': parse_source_target('IsA,' + source, target), 'answer': label}
            data.append(row)

    return data


if __name__ == "__main__":
    with open('beliefbank-data-sep2021/constraints_v2.json', 'r') as f:
        c_graph = json.load(f)

    with open('beliefbank-data-sep2021/silver_facts.json', 'r') as f:
        facts = json.load(f)

    c_data = process_c_graph(c_graph)
    s_data = process_silver_facts(facts)

    with open('beliefbank-data-sep2021/constraints_qa.json', 'w') as f:
        json.dump(c_data, f)

    with open('beliefbank-data-sep2021/silver_qa.json', 'w') as f:
        json.dump(c_data, f)

    with open('beliefbank-data-sep2021/qa.json', 'w') as f:
        json.dump(c_data + s_data, f)




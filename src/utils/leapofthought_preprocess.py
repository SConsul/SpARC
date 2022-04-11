import json
from collections import defaultdict, namedtuple
from tqdm import tqdm

Edge = namedtuple('DataRow', ['question', 'answer', 'source', 'target', 'gold', 'id', 'link_type'])

class DataRow(Edge):
    def __hash__(self):
        return hash(self.question)

    def __eq__(self, other):
        return self.question == other.question


def json_serialize(data):
    return {n: [q._asdict() for q in qs] for n, qs in data.items()}


def parse_question(source, target, predicate):
    s_art = 'an ' if source[0] in {'a', 'e', 'i', 'o', 'u'} else 'a '
    s_art = s_art if source[-1] != 's' else ''
    t_art = 'an ' if target[0] in {'a', 'e', 'i', 'o', 'u'} else 'a '
    t_art = t_art if target[-1] != 's' else ''

    if predicate == '/r/IsA' or predicate == "hypernym":
        return f'Is {s_art}{source} {t_art}{target}?'
    elif predicate == "meronym":
        return f'Does {s_art}{source} have {t_art}{target}?'
    elif predicate == "/r/Antonym":
        return f'{s_art}{source} is the opposite of {t_art}{target}?'
    elif predicate == "/r/CapableOf":
        return f'Can {s_art}{source} {target}?'
    elif predicate == "/r/PartOf":
        return f'Is {s_art}{source} part of {target}?'
    elif predicate == "/r/Desires":
        return f'Does {s_art}{source} desire {target}?'
    else:
        print(predicate)
        raise ValueError("NOOOOO")


def process_constraint(link_info, data_id, link_type):
    source = link_info["subject"]
    target = link_info["object"]
    predicate = link_info["predicate"]
    question = parse_question(source, target, predicate)

    answer = "yes" if link_info["validity"] == "always true" else "no"

    row = DataRow(question=question, answer=answer,
                    source=source, target=target, gold=True,
                    id=data_id, link_type=link_type)
    return row


def get_links(data_dict, data_id, link_type):
    data = []
    info_dict = data_dict[link_type]
    if isinstance(info_dict, list):
        for info in info_dict:
            data.append(process_constraint(info, data_id, link_type))
    else:
        data.append(process_constraint(info_dict, data_id, link_type))

    return data


def process_data(data):
    all_data = defaultdict(list)
    link_types = ["implicit_rule", "property", "statement"]

    for d in tqdm(data):
        metadata = d['metadata']
        data_id = d["id"]
        for link in link_types:
            if link in metadata:
                all_data[data_id].extend(get_links(metadata, data_id, link))

        if "distractors" in metadata:
            for link in link_types:
                if link in metadata['distractors']:
                    all_data[data_id].extend(get_links(metadata['distractors'], data_id, link))
    return all_data
        

if __name__ == "__main__":
    with open('data/hypernyms_training_mix_short_train.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    constraint_data = process_data(data)

    with open('data/leapofthought.json', 'w') as f:
        json.dump(json_serialize(constraint_data), f)
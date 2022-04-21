import json
from tqdm import tqdm
from preprocess_utils import DataRow


def json_serialize(data):
    # return {n: [q._asdict() for q in qs] for n, qs in data.items()}
    output = []
    for info in data:
        output.append(info._asdict())
    return output


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
    # all_data = defaultdict(list)
    all_data = []
    link_types = ["implicit_rule", "property", "statement"]

    for d in tqdm(data):
        metadata = d['metadata']
        data_id = d["id"]
        for link in link_types:
            if link in metadata:
                # all_data[data_id].extend(get_links(metadata, data_id, link))
                all_data.extend(get_links(metadata, data_id, link))

        if "distractors" in metadata:
            for link in link_types:
                if link in metadata['distractors']:
                    # all_data[data_id].extend(get_links(metadata['distractors'], data_id, link))
                    all_data.extend(get_links(metadata['distractors'], data_id, link))
    return all_data
        

if __name__ == "__main__":
    with open('data/original_lot/hypernyms_training_mix_short_train.jsonl', 'r') as f:
        train = [json.loads(line) for line in f]
    
    with open('data/original_lot/hypernyms_training_mix_short_dev.jsonl', 'r') as f:
        val = [json.loads(line) for line in f]

    with open('data/original_lot/hypernyms_statement_only_short_neg_hypernym_rule_test.jsonl') as f:
        test_statement = [json.loads(line) for line in f]
    
    with open('data/original_lot/hypernyms_implicit_only_short_neg_hypernym_rule_test.jsonl') as f:
        test_implicit = [json.loads(line) for line in f]
    
    with open('data/original_lot/hypernyms_explicit_only_short_neg_hypernym_rule_test.jsonl') as f:
        test_explicit = [json.loads(line) for line in f]

    # train_data = process_data(train)
    # val_data = process_data(val)

    test_data_statement = process_data(test_statement)
    test_data_implicit = process_data(test_implicit)
    test_data_explicit = process_data(test_explicit)

    test_data = []
    test_data.extend(test_data_statement)
    test_data.extend(test_data_implicit)
    test_data.extend(test_data_explicit)

    # with open('data/lot_train.json', 'w') as f:
    #     print("Train Data: ", len(train_data)) # 165947
    #     json.dump(json_serialize(train_data), f, indent=1)
    
    # with open('data/lot_val.json', 'w') as f:
    #     print("Val Data: ", len(val_data)) # 7574
    #     json.dump(json_serialize(val_data), f, indent=1)

    # with open('data/lot_test_statement.json', 'w') as f:
    #     print("Test Data: ", len(test_data_statement)) # 4670
    #     json.dump(json_serialize(test_data_statement), f, indent=1)
    
    # with open('data/lot_test_implicit.json', 'w') as f:
    #     print("Test Data: ", len(test_data_implicit)) # 7216
    #     json.dump(json_serialize(test_data_implicit), f, indent=1)
    
    # with open('data/lot_test_explicit.json', 'w') as f:
    #     print("Test Data: ", len(test_data_explicit)) # 9794
    #     json.dump(json_serialize(test_data_explicit), f, indent=1)

    with open('data/lot_test_combined.json', 'w') as f:
        print("Test Data: ", len(test_data)) # 21,680
        json.dump(json_serialize(test_data), f, indent=1)
    
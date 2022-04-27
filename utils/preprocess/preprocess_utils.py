import random
from collections import namedtuple

Edge = namedtuple('DataRow', ['question', 'answer', 'source', 'target', 'gold', 'id', 'link_type', 'pred'],
                  defaults=('', '', ''))


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
             "templates_negated": ["Does [X] not have [Y]?", "Is it true that [X] has not [Y]?"]},
    "/r/IsA": {"templates": ["Is [X] [Y]?"]},
    "hypernym": {"templates": ["Is [X] [Y]?"]},
    "meronym": {"templates": ["Does [X] have [Y]?"]},
    "/r/Antonym": {"templates": ["Is [X] the opposite of [Y]?"]},
    "/r/CapableOf": {"templates": ["Can [X] [Y]?"]},
    "/r/PartOf": {"templates": ["Is [X] part of [Y]?"]},
    "/r/Desires": {"templates": ["Does [X] desire [Y]?"]}
}


def parse_source_target(source, target, non_countable, use_pos=True):
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


def json_serialize_adj_list(data):
    return {n: [q._asdict() for q in qs] for n, qs in data.items()}


def json_serialize(data):
    return [q._asdict() for q in data]


def flatten(l):
    return [x for sub_l in l for x in sub_l]

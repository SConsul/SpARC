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
             "templates_negated": ["Does [X] not have [Y]?", "Is it true that [X] has not [Y]?"]}
}


def json_serialize(data):
    return {n: [q._asdict() for q in qs] for n, qs in data.items()}

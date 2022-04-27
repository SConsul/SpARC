from collections import defaultdict
from utils.preprocess.preprocess_utils import DataRow, parse_source_target


def traverse(data_adj_list):
    visited = defaultdict(set)

    def _traverse_dfs(source, n):
        for e in data_adj_list.get(n):
            # "Groundtruth" answer is expected model prediction for this multihop edge
            s_n_edge = DataRow(question=parse_source_target(source, e.target, non_countable, use_random=False),
                               answer=e.pred, source=source, target=e.target, gold=False,
                               id=e.id, link_type=e.link_type)

            if s_n_edge not in visited[source]:
                visited[source].add(s_n_edge)

                # Don't go down 'no' edges
                if s_n_edge.target in data_adj_list and e.pred == 'yes':
                    _traverse_dfs(source, s_n_edge.target)
        return

    # Only create multihop questions (avoid single hop)
    for n, edges in data_adj_list.items():
        for e in edges:
            if e.target in data_adj_list and e.pred == 'yes':
                _traverse_dfs(n, e.target)

    return visited


def gen_belief_graph(inf_out):
    skipped = 0
    adj_list = defaultdict(set)
    for data_row in inf_out:
        data_row = DataRow(**data_row)
        # assert data_row.pred != '', "Need model predictions on these edges"

        if data_row.pred in {'yes', 'no'}:
            adj_list[data_row.source].add(data_row)
        else:
            skipped += 1

    print(f"Skipped edges for consistency: {skipped}")
    multihop_questions = traverse(adj_list)
    return adj_list, multihop_questions


def eval_consistency(multihop_out):
    violations = 0
    num_edges = 0
    for e in multihop_out:
        if type(e) is dict:
            e = DataRow(**e)

        assert e.pred != '', "Need model predictions on these edges"
        if e.answer != e.pred:
            violations += 1
        num_edges += 1

    return 1-(violations/num_edges)


with open('./beliefbank-data-sep2021/non_countable.txt', 'r') as f:
    non_countable = f.read()

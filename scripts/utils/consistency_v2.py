import json
import argparse
import numpy as np


def create_graph(pred_results_file, c_graph_path, q_consistency_path):
    """
    Returns predicted results in the form of the adjacency matrix:
    A[source_node_id][target_node_id] = predicted_value
    :param pred_results_file: path to json file that has results of inference
    :param c_graph_path: path to consistency graph json file (original dataset)
    :param q_consistency_path: path to file mapping question -> source, target
    :return:
    """
    with open(c_graph_path, 'r') as f:
        c_graph = json.load(f)

    nodes = {n['id']: i for i, n in enumerate(c_graph['nodes'])}

    with open(q_consistency_path, 'r') as f:
        q_consistency = json.load(f)

    q_to_nodes = {q['question']: (q['source'], q['target']) for q in q_consistency}

    with open(pred_results_file, 'r') as f:
        pred_results = json.load(f)

    pred = np.zeros((len(nodes), len(nodes)))
    for pred in pred_results:
        question = pred['q']
        source, target = q_to_nodes[question]
        s, t = nodes[source], nodes[target]
        answer = pred['pred'].split()[2]
        pred[s][t] = (answer == 'yes' or answer == 'Yes')

    return pred


def consistency(results_path, qa_to_node_path, q_consistency_path):
    A_single = create_graph(results_path, qa_to_node_path, q_consistency_path)
    num = 0.0
    den = 0.0
    A_multi = A_single
    for _ in range(2, 10):
        A_multi = A_multi @ A_single
        num += np.sum((A_multi > 0).astype(float) * A_single)
        den += np.sum((A_multi > 0).astype(float))

    return num / den


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_graph', default='./beliefbank-data-sep2021/constraints_v2.json')
    parser.add_argument('--q_consistency', default='./beliefbank-data-sep2021/qa_consistency.json')
    parser.add_argument('--results_path', default="./beliefbank-data-sep2021/baseline.json")
    args = parser.parse_args()

    consis = consistency(args.results_path, args.c_graph, args.q_consistency)
    print("Consistency = ", consis)

import json
import argparse
import numpy as np


def create_graph(pred_results_file, c_graph):
    """
    Returns predicted results in the form of the adjacency matrix:
    A[source_node_id][target_node_id] = predicted_value

    Inputs: 
     (string) pred_results_file: path to json file that has results of inference
     (string) qa_to_nodes_file: path to json file that maps each node to an (int) id
     (int) num_nodes: the total number of unique nodes
    """
    with open(c_graph, 'r') as f:
        c_graph = json.load(f)

    nodes = {n['id']: i for i, n in enumerate(c_graph['nodes'])}

    with open(pred_results_file, 'r') as f:
        pred_results = json.load(f)

    pred = np.zeros((len(nodes), len(nodes)))
    for pred in pred_results:
        question = pred['q']
        s = nodes[question['source']]
        t = nodes[question['target']]
        answer = pred['pred'].split()[2]
        pred[s][t] = (answer == 'yes' or answer == 'Yes')

    return pred


def consistency(results_path, qa_to_node_path):
    A_single = create_graph(results_path, qa_to_node_path)
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
    parser.add_argument('--results_path', default="./beliefbank-data-sep2021/baseline.json")
    args = parser.parse_args()

    consis = consistency(args.results_path, args.c_graph)
    print("Consistency = ", consis)

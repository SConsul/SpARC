import json
import argparse
import numpy as np

def create_graph(pred_results_file, qa_to_nodes_file, num_nodes):
    """
    Returns predicted results in the form of the adjacency matrix:
    A[source_node_id][target_node_id] = predicted_value

    Inputs: 
     (string) pred_results_file: path to json file that has results of inference
     (string) qa_to_nodes_file: path to json file that maps each node to an (int) id
     (int) num_nodes: the total number of unique nodes
    """
    with open(qa_to_nodes_file, 'r') as f:
        qa_to_node = json.load(f)

    with open(pred_results_file, 'r') as f:
        pred_results = json.load(f)
    
    pred = np.zeros((num_nodes,num_nodes))
    for pred in pred_results:
        question = pred['q']
        source = qa_to_node[question]['source']
        target = qa_to_node[question]['target']
        answer = pred['pred'].split()[2]
        pred[source][target] = (answer == 'yes' or answer =='Yes')
    
    return pred
    
def consistency(results_path,qa_to_node_path,num_nodes):
    A_single = create_graph(results_path,qa_to_node_path,num_nodes)
    num = 0.0
    den = 0.0
    A_multi = A_single@A_single
    num += np.sum(A_multi*A_single)
    den += np.sum(A_single)

    for _ in range(3,10):
        A_multi = A_multi@A_single 
        num += np.sum(A_multi*A_single)
        den += np.sum(A_multi)

    return num/den

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', default="../beliefbank-data-sep2021/baseline.json")
    parser.add_argument('--question_to_node_file', default="../beliefbank-data-sep2021/qa_to_nodes.json")
    parser.add_argument('--num_nodes', type = int, default=2000)
    args = parser.parse_args()

    consis = consistency(args.results_path, args.question_to_node_file, args.num_nodes)
    print("Consistency = ", consis)
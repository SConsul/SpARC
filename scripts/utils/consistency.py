import json
import argparse
import numpy as np


def create_constraint_graph(qa_file):
    """
    Use QA json file to create an adjacency list of the form:
    {
        source_node: [[target_node, truth_value], [target_node, truth_value], ...]
    }
    """
    with open(qa_file, 'r') as f:
        qa_edges = json.load(f)
    
    constraint_graph = {}

    for edge in qa_edges:
        source_node = edge['source']
        target_node = edge['target']
        truth_value = 1 if edge['answer'] == 'yes' else 0

        if source_node not in constraint_graph:
            constraint_graph[source_node] = []

        if target_node not in constraint_graph:
            constraint_graph[target_node] = []

        constraint_graph[source_node].append([target_node, truth_value])
    
    return constraint_graph


def create_constraint_graph_from_constraint_file(constraint_file):
    """
    Use constraint json file to create an adjacency list of the form:
    {
        source_node: [[target_node, truth_value], [target_node, truth_value], ...]
    }
    """
    with open(constraint_file, 'r') as f:
        constraint_graph = json.load(f)

    nodes = constraint_graph['nodes']
    edges = constraint_graph['links']
    constraint_graph = {}

    for node in nodes:
        constraint_graph[node['id']] = []
    
    for edge in edges:
        source_node = edge['source'] if edge['direction'] == 'forward' else edge['target']
        target_node = edge['target'] if edge['direction'] == 'forward' else edge['source']
        truth_value = 1 if edge['weight'] == 'yes_yes' else 0

        constraint_graph[source_node].append([target_node, truth_value])
    
    return constraint_graph


def get_node_truth(node, constraint_graph, visited, truth_values):
    """
    Get all nodes in subtree and their truth values
    """
    if node not in visited:
        visited.add(node)

        for target_node in constraint_graph[node]:
            if (target_node[0] not in truth_values.keys()):
                truth_values[target_node[0]] = 1 if target_node[1] == 1 and truth_values[node] == 1 else 0          
            truth_values = get_node_truth(target_node[0], constraint_graph, visited, truth_values)
    
    return truth_values


def create_node_truth_graph(nodes, constraint_graph):
    """
    Create node consistency truth graph of form:
    {
        source_node: {{target_node: consistent_truth_value}, ...}
    }
    """
    node_truths = {}
    for node in nodes:
        if node in constraint_graph:
            visited = set()
            truth_values = {}
            truth_values[node] = 1
            node_truths[node] = get_node_truth(node, constraint_graph, visited, truth_values)
        
    return node_truths


def process_pred_results(pred_results_file, qa_to_nodes_file):
    """
    Returns predicted results in the form:
    [(source_node, target_node, predicted_value), ...]
    """
    with open(qa_to_nodes_file, 'r') as f:
        qa_to_node = json.load(f)

    with open(pred_results_file, 'r') as f:
        pred_results = json.load(f)
    
    processed_pred = []
    
    for pred in pred_results:
        question = pred['q']
        source = qa_to_node[question]['source']
        target = qa_to_node[question]['target']
        answer = pred['pred'].split()[2]
        processed_pred.append((source, target, answer))
    
    return processed_pred


def consistency(true_constraint_file, pred_results_file, qa_to_nodes_file):
    """
    Calculate consistency
    """
    true_constraint_graph = create_constraint_graph(true_constraint_file)
    node_truth_graph = create_node_truth_graph(true_constraint_graph.keys(), true_constraint_graph)
    pred_results = process_pred_results(pred_results_file, qa_to_nodes_file)
    consistencies = []
    num_violated = 0
    num_applicable = len(pred_results)

    for result in pred_results:
        source_node, target_node, predicted_truth = result
        predicted_truth = 1 if predicted_truth.lower() == 'yes' else 0
        if target_node in node_truth_graph[source_node]:
            num_violated += 1 if node_truth_graph[source_node][target_node] != predicted_truth else 0
        else:
            num_violated = 0

    # T = |violated_constraints| / |applicable_constraints|
    inconsistency = float(num_violated) / float(num_applicable)
    consistency = 1 - inconsistency
    
    return consistency 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--consistency_graph_path', default="./beliefbank-data-sep2021/qa.json")
    parser.add_argument('--results_path', default="./beliefbank-data-sep2021/baseline.json")
    parser.add_argument('--question_to_node_file', default="./beliefbank-data-sep2021/qa_to_nodes.json")
    args = parser.parse_args()

    # Evaluate consistency
    consis = consistency(args.consistency_graph_path, args.results_path, args.question_to_question_to_node_filenode_file)
    print("Consistency = ", consis)
    
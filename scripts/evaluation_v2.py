import json
from operator import truth


def create_constraint_graph(constraint_file):
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


def get_node_truth(node, constraint_graph, target_nodes, truth_values):
    """
    Count the number of nodes that can be reached from a source node
    """
    if node not in target_nodes:
        target_nodes.add(node)

        for target_node in constraint_graph[node]:
            if (target_node[0] not in truth_values.keys()):
                truth_values[target_node[0]] = 1 if target_node[1] == 1 and truth_values[node] == 1 else 0          
            target_nodes, truth_values = get_node_truth(target_node[0], constraint_graph, target_nodes, truth_values)
    
    return target_nodes, truth_values
    

def eval_consistency(sentences, constraint_file):
    constraint_graph = create_constraint_graph(constraint_file)
    consistencies = []
    applicable_constraints = {}
    for sentence in sentences:
        source_node = sentence['source']
        target_node = sentence['target']
        answer = sentence['answer']

        s_nodes = set()
        s_truth = {}
        t_nodes = set()
        t_truth = {}
        if source_node not in applicable_constraints.keys():
            s_truth[source_node] = 1
            s_nodes, s_truth = get_node_truth(source_node, constraint_graph, s_nodes, s_truth)
            
        if target_node not in applicable_constraints.keys():
            t_truth[target_node] = 1
            t_nodes, t_truth = get_node_truth(target_node, constraint_graph, t_nodes, t_truth)
        
        intersect = s_nodes.intersection(t_nodes)
        s_truth_intersect = dict((key, s_truth[key]) for key in intersect if key in s_truth.keys())
        t_truth_intersect = dict((key, t_truth[key]) for key in intersect if key in t_truth.keys())

        num_applicable_contraints = len(intersect)
        num_violated_constraints = 0
        for node in intersect:
            num_violated_constraints += 0 if s_truth_intersect[node] == t_truth_intersect[node] else 1
        
        # T = |violated_constraints| / |applicable_constraints|
        inconsistency = float(num_violated_constraints) / float(num_applicable_contraints)
        consistency = 1 - inconsistency
        consistencies.append(consistency)
    
    return consistencies

if __name__ == "__main__":
    sentences = [{'source': 'IsA,palm tree', 'target': 'IsA,fire', 'answer': "yes"}, {'source': 'IsA,palm tree', 'target': 'IsA,computer', 'answer': "yes"}, {'source': 'IsA,bird', 'target': 'IsA,deer', 'answer': "yes"}, {'source': 'IsA,fire', 'target': 'IsA,bird', 'answer': "yes"}, {'source': 'IsA,bird', 'target': 'IsA,bird', 'answer': "yes"}]
    constraint_file = 'beliefbank-data-sep2021/constraints_v2.json'
    
    consistencies = eval_consistency(sentences, constraint_file)
    print(consistencies)

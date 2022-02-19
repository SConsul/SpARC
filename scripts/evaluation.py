import json


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


def get_node_count(node, constraint_graph, target_nodes, node_count):
    """
    Count the number of nodes that can be reached from a source node
    """
    if node not in target_nodes:
        node_count += 1
        target_nodes.add(node)

        for target_node in constraint_graph[node]:
            node_count, target_nodes = get_node_count(target_node[0], constraint_graph, target_nodes, node_count)
    
    return node_count, target_nodes


def get_nodes_path(source, target, constraint_graph, visited, marked, base_path):
    """
    Get all nodes on path from source to target node
    """ 
    if source not in visited:
        visited.add(source)

        if (source == target):
            for node in base_path:
                marked.add(node)
            return
        
        for target_node in constraint_graph[source]:
            base_path.append(target_node[0])
            get_nodes_path(target_node[0], target, constraint_graph, visited, marked, base_path)
            if (base_path[-1] in marked):
                marked.add(source)
            base_path.pop()


def eval_consistency(sentences, constraint_file):
    constraint_graph = create_constraint_graph(constraint_file)
    consistencies = []
    applicable_constraints = {}
    for sentence in sentences:
        source_node = sentence['source']
        target_node = sentence['target']

        if source_node not in applicable_constraints.keys():
            target_nodes = set()
            num_applicable_contraints = 0
            num_applicable_contraints, target_nodes = get_node_count(source_node, constraint_graph, target_nodes, num_applicable_contraints)
            applicable_constraints[source_node] = num_applicable_contraints
        
        visited = set()
        marked = set()
        base_path = [source_node]
        get_nodes_path(source_node, target_node, constraint_graph, visited, marked, base_path)
        num_violated_constraints = len(marked)

        # T = |violated_constraints| / |applicable_constraints|
        inconsistency = float(num_violated_constraints) / float(applicable_constraints[source_node])
        consistency = 1 - inconsistency
        consistencies.append(consistency)
    
    return consistencies

if __name__ == "__main__":
    sentences = [{'source': 'IsA,palm tree', 'target': 'IsA,fire', 'answer': "yes"}, {'source': 'IsA,palm tree', 'target': 'IsA,computer', 'answer': "yes"}, {'source': 'IsA,palm tree', 'target': 'IsA,palm tree', 'answer': "yes"}]
    constraint_file = 'beliefbank-data-sep2021/constraints_v2.json'
    
    consistencies = eval_consistency(sentences, constraint_file)
    print(consistencies)
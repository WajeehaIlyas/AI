class TreeNode:
    def __init__(self, node_id, value=None, children=None):
        self.node_id = node_id 
        self.value = value 
        self.children = children if children is not None else []  

root = TreeNode(0, children=[
    TreeNode(1, children=[
        TreeNode(3, value=3),  
        TreeNode(4, value=6)  
    ]),
    TreeNode(2, children=[
        TreeNode(5, value=9),  
        TreeNode(6, value=2)   
    ])
])

minimax_iterations = 0

def minimax(node, is_maximizing, path):
    global minimax_iterations
    minimax_iterations += 1 

    if not node.children:
        return node.value, path + [f"Node {node.node_id} ({node.value})"]

    if is_maximizing:
        max_value = float('-inf')
        best_path = []
        for child in node.children:
            val, child_path = minimax(child, False, path + [f"Node {node.node_id}"])
            if val > max_value:
                max_value = val
                best_path = child_path
        return max_value, best_path

    else:  
        min_value = float('inf')
        best_path = []
        for child in node.children:
            val, child_path = minimax(child, True, path + [f"Node {node.node_id}"])
            if val < min_value:
                min_value = val
                best_path = child_path
        return min_value, best_path

minimax_iterations = 0  
optimal_value, optimal_path = minimax(root, True, [])

print("Optimal value found using Minimax:", optimal_value)
print("Total nodes evaluated in Minimax:", minimax_iterations)
print("Path taken:", " -> ".join(optimal_path))


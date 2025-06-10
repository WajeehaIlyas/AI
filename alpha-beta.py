import math

#game tree
game_tree = {
    "A": ["B", "C"],
    "B": ["D", "E", "F"],
    "C": ["G", "H", "I"],
    "D": ["J", "K"],
    "E": ["L", "M"],
    "F": ["N", "O"],
    "G": ["P", "Q"],
    "H": ["R", "S"],
    "I": ["T", "U"],
    "J": 10, "K": 20, "L": 30, "M": 15, "N": 15, "O": 30, 
    "P": 25, "Q": 35, "R": 5, "S": 10, "T": 35, "U": 40
}

def alphabeta(node, alpha, beta, is_maximizing):
    global nodes_evaluated, pruned_nodes
    nodes_evaluated += 1  

    if node in game_tree and isinstance(game_tree[node], int):
        return game_tree[node]

    if is_maximizing:
        best_value = float('-inf')
        for child in game_tree[node]:
            value = alphabeta(child, alpha, beta, False)
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            if beta <= alpha:
                pruned_nodes += 1  
                break  
        return best_value
    else:
        best_value = float('inf')
        for child in game_tree[node]:
            value = alphabeta(child, alpha, beta, True)
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if beta <= alpha:
                pruned_nodes += 1
                break  
        return best_value

nodes_evaluated = 0
pruned_nodes = 0

optimal_value = alphabeta("A", float('-inf'), float('inf'), True)
print("Optimal value found using Alpha-Beta Pruning:", optimal_value)
print("Total nodes evaluated:", nodes_evaluated)
print("Number of pruned nodes:", pruned_nodes)

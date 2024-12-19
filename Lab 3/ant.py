import random

class AntColonyOptimization:
    def __init__(self, graph, source, destination, n_ants, n_iterations, alpha, beta, evaporation_rate, pheromone_constant):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_constant = pheromone_constant

        # Initialize pheromone levels on all edges
        self.pheromone = {edge: 1.0 for edge in graph}

    def _calculate_probability(self, current_node, neighbors):
        probabilities = {}
        denominator = 0
        for neighbor in neighbors:
            edge = (current_node, neighbor)
            if edge in self.graph:
                heuristic = 1 / self.graph[edge]  # Heuristic is inverse of edge weight (e.g., distance)
                numerator = (self.pheromone[edge] ** self.alpha) * (heuristic ** self.beta)
                probabilities[neighbor] = numerator
                denominator += numerator

        # Normalize probabilities
        for neighbor in probabilities:
            probabilities[neighbor] /= denominator

        return probabilities

    def _select_next_node(self, probabilities):
        nodes = list(probabilities.keys())
        probabilities_list = list(probabilities.values())
        return random.choices(nodes, probabilities_list, k=1)[0]

    def _construct_path(self):
        current_node = self.source
        path = [current_node]
        visited = set()
        while current_node != self.destination:
            visited.add(current_node)
            neighbors = [node for node in self.graph if node[0] == current_node and node[1] not in visited]
            if not neighbors:
                return None  # Dead-end reached
            probabilities = self._calculate_probability(current_node, [edge[1] for edge in neighbors])
            next_node = self._select_next_node(probabilities)
            path.append(next_node)
            current_node = next_node
        return path

    def _update_pheromone(self, paths):
        # Evaporate existing pheromone
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

        # Add pheromone based on paths
        for path, path_length in paths:
            pheromone_contribution = self.pheromone_constant / path_length
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                self.pheromone[edge] += pheromone_contribution

    def optimize(self):
        for iteration in range(self.n_iterations):
            paths = []
            for _ in range(self.n_ants):
                path = self._construct_path()
                if path:
                    path_length = sum(self.graph[(path[i], path[i + 1])] for i in range(len(path) - 1))
                    paths.append((path, path_length))
            self._update_pheromone(paths)

        # Find the best path based on pheromone levels
        best_path = None
        best_path_length = float('inf')
        for path, path_length in paths:
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        return best_path, best_path_length


# Example graph: (node1, node2): weight
graph = {
    (0, 1): 2, (0, 2): 2, (1, 2): 1,
    (1, 3): 1, (2, 3): 2, (2, 4): 3,
    (3, 4): 1
}

# Parameters
source = 0
destination = 4
n_ants = 10
n_iterations = 100
alpha = 1.0  # Importance of pheromone
beta = 2.0   # Importance of heuristic (distance)
evaporation_rate = 0.1
pheromone_constant = 100.0

# Run ACO
aco = AntColonyOptimization(graph, source, destination, n_ants, n_iterations, alpha, beta, evaporation_rate, pheromone_constant)
best_path, best_path_length = aco.optimize()

print("Best path:", best_path)
print("Path length:", best_path_length)

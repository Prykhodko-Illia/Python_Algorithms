import random
from typing import List, Tuple
from algorithms import dijkstra, bellman_ford, floyd_warshall, johnson, prim, kruskal

Edge = Tuple[int, int, int]
IntList = List[int]
Tree = List[Edge]
Matrix = List[List[int]]

class Graph:
    def __init__(self, vertices_number: int, density: float, max_weight: int, directed: bool):
        self.vertices: IntList = list(range(1, vertices_number + 1))
        self.edges: Tree = []
        self.vertices_count: int = vertices_number
        self.directed: bool = directed

        for i in range(1, vertices_number + 1):
            for j in range(1, vertices_number + 1):
                if i != j:
                    if random.random() <= density:
                        weight = random.randint(1, max_weight)
                        self.edges.append((i, j, weight))
                        if not directed:
                            self.edges.append((j, i, weight))

    def get_matrix(self) -> Matrix:
        n = len(self.vertices)
        result = [[float('inf')] * n for _ in range(n)]

        for i in range(n):
            result[i][i] = 0

        for edge in self.edges:
            u = edge[0] - 1
            v = edge[1] - 1
            weight = edge[2]
            result[u][v] = weight
            if not self.directed:
                result[v][u] = weight

        return result

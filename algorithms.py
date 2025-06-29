import heapq
from collections import defaultdict
from typing import List, Tuple
from utils import find, unite

Edge = Tuple[int, int, int]
IntList = List[int]
Tree = List[Edge]
Matrix = List[List[int]]


def dijkstra(graph, start: int, final: int) -> IntList:
    n = len(graph.vertices)
    if start < 1 or start > n or final < 1 or final > n:
        return []

    INF = float('inf')
    distance = [INF] * (n + 1)
    previous = [None] * (n + 1)
    visited = [False] * (n + 1)

    distance[start] = 0
    pq = []
    heapq.heappush(pq, (0, start))

    while pq:
        dist_u, u = heapq.heappop(pq)

        if visited[u]:
            continue
        visited[u] = True

        if u == final:
            break

        for from_node, to_node, weight in graph.edges:
            if from_node == u:
                v = to_node
            elif to_node == u:
                v = from_node
            else:
                continue

            if not visited[v] and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                previous[v] = u
                heapq.heappush(pq, (distance[v], v))

    path = []
    if distance[final] == INF:
        return []

    v = final
    while v is not None:
        path.append(v)
        v = previous[v]

    path.reverse()
    return path


def bellman_ford(graph, start: int, final: int) -> IntList:
    n = len(graph.vertices)
    if start < 1 or start > n or final < 1 or final > n:
        return []

    INF = float('inf')
    distance = [INF] * (n + 1)
    previous = [None] * (n + 1)
    distance[start] = 0

    for _ in range(n - 1):
        for u, v, weight in graph.edges:
            if distance[u] != INF and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                previous[v] = u
            if distance[v] != INF and distance[v] + weight < distance[u]:
                distance[u] = distance[v] + weight
                previous[u] = v

    for u, v, weight in graph.edges:
        if distance[u] != INF and distance[u] + weight < distance[v]:
            return []
        if distance[v] != INF and distance[v] + weight < distance[u]:
            return []

    path = []
    if distance[final] == INF:
        return []

    v = final
    while v is not None:
        path.append(v)
        v = previous[v]

    path.reverse()
    return path


def bellman_ford_directed(graph, start: int, final: int) -> IntList:
    n = len(graph.vertices)
    if start < 1 or start > n or final < 1 or final > n:
        return []

    INF = float('inf')
    distance = [INF] * (n + 1)
    previous = [None] * (n + 1)
    distance[start] = 0

    for _ in range(n - 1):
        for u, v, weight in graph.edges:
            if distance[u] != INF and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                previous[v] = u

    for u, v, weight in graph.edges:
        if distance[u] != INF and distance[u] + weight < distance[v]:
            return []

    path = []
    if distance[final] == INF:
        return []

    v = final
    while v is not None:
        path.append(v)
        v = previous[v]

    path.reverse()
    return path


def floyd_warshall(graph, start: int, final: int) -> IntList:
    n = len(graph.vertices)
    if start < 1 or start > n or final < 1 or final > n:
        return []

    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[-1] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, weight in graph.edges:
        u_idx, v_idx = u - 1, v - 1
        dist[u_idx][v_idx] = weight
        dist[v_idx][u_idx] = weight
        next_node[u_idx][v_idx] = v_idx
        next_node[v_idx][u_idx] = u_idx

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF and dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    start_idx, final_idx = start - 1, final - 1
    if dist[start_idx][final_idx] == INF:
        return []

    path = []
    current = start_idx
    while current != final_idx:
        path.append(current + 1)
        if next_node[current][final_idx] == -1:
            return []
        current = next_node[current][final_idx]
    path.append(final)

    return path


def bellman_ford_distances(graph, start: int) -> list:
    n = len(graph.vertices)
    INF = float('inf')
    distance = [INF] * (n + 1)
    distance[start] = 0

    for _ in range(n - 1):
        for u, v, weight in graph.edges:
            if distance[u] != INF and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
            if distance[v] != INF and distance[v] + weight < distance[u]:
                distance[u] = distance[v] + weight

    for u, v, weight in graph.edges:
        if distance[u] != INF and distance[u] + weight < distance[v]:
            return []
        if distance[v] != INF and distance[v] + weight < distance[u]:
            return []

    return distance


def dijkstra_johnson(edges, start: int, final: int, n: int) -> IntList:
    from collections import defaultdict
    import heapq

    adj = defaultdict(list)
    for u, v, weight in edges:
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    INF = float('inf')
    distance = [INF] * (n + 1)
    previous = [None] * (n + 1)
    distance[start] = 0
    pq = [(0, start)]

    while pq:
        dist_u, u = heapq.heappop(pq)
        if dist_u > distance[u]:
            continue
        if u == final:
            break
        for v, weight in adj[u]:
            new_dist = distance[u] + weight
            if new_dist < distance[v]:
                distance[v] = new_dist
                previous[v] = u
                heapq.heappush(pq, (new_dist, v))

    if distance[final] == INF:
        return []

    path = []
    v = final
    while v is not None:
        path.append(v)
        v = previous[v]
    return path[::-1]


def johnson(graph, start: int, final: int) -> IntList:
    n = len(graph.vertices)
    if start < 1 or start > n or final < 1 or final > n:
        return []

    class TempGraph:
        def __init__(self, vertices_count, edges):
            self.vertices = list(range(vertices_count + 1))
            self.edges = edges

    temp_edges = graph.edges + [(n + 1, i, 0) for i in range(1, n + 1)]
    temp_graph = TempGraph(n + 1, temp_edges)

    h = bellman_ford_distances(temp_graph, n + 1)
    if not h:
        return []

    reweighted_edges = []
    for u, v, w in graph.edges:
        new_weight = w + h[u] - h[v]
        reweighted_edges.append((u, v, new_weight))

    return dijkstra_johnson(reweighted_edges, start, final, n)


def prim(graph) -> Tree:
    n = len(graph.vertices)
    visited = [False] * (n + 1)
    result = []

    adj = defaultdict(list)
    for u, v, weight in graph.edges:
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    pq = []
    start = 1
    visited[start] = True

    for v, weight in adj[start]:
        heapq.heappush(pq, (weight, start, v))

    while pq and len(result) < n - 1:
        weight, u, v = heapq.heappop(pq)

        if visited[v]:
            continue

        visited[v] = True
        result.append((u, v, weight))

        for next_v, next_weight in adj[v]:
            if not visited[next_v]:
                heapq.heappush(pq, (next_weight, v, next_v))

    return result


def kruskal(graph) -> Tree:
    parent = list(range(len(graph.vertices) + 1))
    rank = [0] * (len(graph.vertices) + 1)
    result = []

    sorted_edges = sorted(graph.edges, key=lambda edge: edge[2])

    for u, v, w in sorted_edges:
        if find(u, parent) != find(v, parent):
            unite(u, v, parent, rank)
            result.append((u, v, w))
            if len(result) == len(graph.vertices) - 1:
                break

    return result

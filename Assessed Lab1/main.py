from typing import List, Tuple, Union, Dict
import math
import sys
from timeit import default_timer

class PriorityQueue:
    def __init__(self, size: int):
        # Create a NumPy array to store (priority, item) pairs
        self.elements: List[Tuple[int, int]] = [(0,0)] * size
        self.capacity: int = size
        self.size: int = 0  # Track current number of elements in the queue

    # Check if the priority queue is empty
    def is_empty(self) -> bool:
        return self.size == 0

    # Add an element with a priority
    def push(self, priority: int, item: int):
        if self.size >= self.capacity:
            raise RuntimeError("Priority queue is full")
        
        self.elements[self.size] = (priority, item)
        self.size += 1
        self.sift_up(self.size - 1)

    # Remove and return the elements with the highest priority (smallest value)
    def pop(self) -> int:
        if self.is_empty():
            raise RuntimeError("Priority queue is empty")

        # Swap the first element with the last element
        self.elements[0], self.elements[self.size - 1] = self.elements[self.size - 1], self.elements[0]
        item = self.elements[self.size - 1][1]  # Get the item to return
        self.size -= 1  # Reduce size
        if not self.is_empty():
            self.sift_down(0)
        return item

    # Helper to maintain the heap property when inserting into the queue
    def sift_up(self, index: int):
        parent = (index - 1) // 2
        if index > 0 and self.elements[index][0] < self.elements[parent][0]:
            self.elements[index], self.elements[parent] = self.elements[parent], self.elements[index]
            self.sift_up(parent)

    # Helper to maintain the heap property when removing elements from the queue
    def sift_down(self, index: int):
        smallest: int = index
        left: int = 2 * index + 1
        right: int = 2 * index + 2

        if left < self.size and self.elements[left][0] < self.elements[smallest][0]:
            smallest = left
        if right < self.size and self.elements[right][0] < self.elements[smallest][0]:
            smallest = right

        if smallest != index:
            self.elements[index], self.elements[smallest] = self.elements[smallest], self.elements[index]
            self.sift_down(smallest)


# CustomSet mimics a set using a dictionary
class CustomSet:
    def __init__(self):
        self.items: Dict = {}  # Dictionary for storing items. Used as an abstract set type
        self.count: int = 0    # Number of items
    
    # Add an item to the set
    def add(self, item: Union[int, float, str]) -> None:
        if not self.contains(item):
            self.items[item] = item
            self.count += 1

    # Remove an item from the set
    def remove(self, item: Union[int, float, str]) -> None:
        if self.contains(item):
            del self.items[item]
            self.count -= 1

    # Print items to the set
    def printItems(self):
        print(self.items)

    # Check if an item is in the set
    def contains(self, item: Union[int, float, str]) -> bool:
        return item in self.items
        

    # Python's special method to check membership (i.e., 'in' operator)
    def __contains__(self, item: Union[int, float, str]) -> bool:
        return self.contains(item)

    # Get the size of the set
    def get_size(self) -> int:
        return self.count


# Calculate the Euclidean distance between two 2D points
def euclidean_distance(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)


# Read input from file and parse graph and vertices
def read_file(filename: str) -> Union[Tuple[List[List[Tuple[int, int]]], List[Tuple[float, float]], int, int], None]:
    with open(filename, 'r') as file:
        try:
            # Read number of vertices (n) and edge (m)
            line = file.readline()
            n, m = int(line.split()[0]), int(line.split()[1])
            vertices = []
            
            # Read the coordinates of each vertex
            for _ in range(n):
                line = file.readline().split()
                vertex = (float(line[1]), float(line[2]))
                vertices.append(vertex)

            # Initialise the graph with an empty adjacency list
            graph: List[List[Tuple[int, int]]] = [[] for _ in range(m)]
            
            # Read the edges and build the adjacency list
            for _ in range(m):
                line = file.readline().strip()
                if line:  # Ensure the line is not empty
                    parts = line.split()
                    if len(parts) != 3:  # Expecting start, end, weight
                        print("Invalid edge entry:", line)
                        continue

                    # Read start and goal vertices
                    start = int(parts[0])  - 1  # Convert to 0-based index
                    end = int(parts[1]) - 1     # Convert to 0-based index
                    weight = int(parts[2])
                    graph[start].append((end, weight))

            # Read start and goal vertex
            line = file.readline().strip()
            if line:
                parts = line.split()
                if len(parts) != 2:
                    print("Invalid start/goal entry:", line)
                    return None
                start_vertex = int(parts[0]) - 1  # Convert to 0-based index
                goal_vertex = int(parts[1]) - 1    # Convert to 0-based index
                return graph, vertices, start_vertex, goal_vertex
        except FileNotFoundError:
            print("Could not open file.", file=sys.error)

    print("File reading completed.")
    return graph, vertices, start_vertex, goal_vertex


# Dijkstra's algorithm to find the shortest path
def dijkstra(graph: List[List[Tuple[int, int]]], start_vertex: int, goal_vertex: int) -> Tuple[List[int], float]:
    num_vertices: int = len(graph)

    # Initialize distances to infinity and parents to -1 using np.full
    distances: List[float] = [float('inf')] * num_vertices
    parent: List[int] = [-1] * num_vertices
    distances[start_vertex] = 0

    pq: PriorityQueue = PriorityQueue(num_vertices)
    pq.push(0, start_vertex)

    while not pq.is_empty():
        current_vertex: int = pq.pop()

        if current_vertex == goal_vertex:
            break  # Stop if the goal vertex is reached

        # Explore all neighbours of the current vertex
        for neighbor, weight in graph[current_vertex]:
            new_distance: float = distances[current_vertex] + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                parent[neighbor] = current_vertex
                pq.push(new_distance, neighbor)

    # Reconstruct the shortest path
    path: List[int] = []
    current = goal_vertex
    
    while current != -1:
        path.append(current + 1)  # Convert back to 1-based index
        current: int = parent[current]

    path.reverse()  # Reverse the path to get it from start to goal
    return path, distances[goal_vertex]


# DFS-based algorithm to find the longest path in a graph
def dfs(graph: List[List[Tuple[int, int]]], current_vertex: int, goal_vertex: int, visited: CustomSet, path: List, max_path: List, max_length: int, current_length: int):
    visited.add(current_vertex)  # Mark the current vertex as visited
    path.append(current_vertex)

    # If we reached the goal vertex and the current path length is greater than max_length
    if current_vertex == goal_vertex:
        # Update max_length and max_path if we found a longer path
        if current_length > max_length[0]: 
            max_length[0] = current_length
            max_path.clear()  # Clear the previous max path
            max_path.extend(path)  # Copy the current path to max_path
    else:
        # Explore all unvisited neighbours of the current vertex
        for neighbor, weight in graph[current_vertex]:
            if neighbor not in visited:  # Only visit unvisited neighbors
                dfs(graph, neighbor, goal_vertex, visited, path, max_path, max_length, current_length + weight)

    # Backtrack: remove current vertex from path and mark it as unvisited
    path.pop()
    visited.remove(current_vertex)


# Find the longest path from start to goal using DFS
def find_longest_path(graph: List[List[Tuple[int, int]]], start_vertex: int, goal_vertex: int) -> Tuple[List[int], int]:
    visited: CustomSet = CustomSet()  # Use a set for visited nodes for efficient lookups
    max_path: List[int] = []
    max_length: List[int] = [0]  # Store the max path length in a mutable list
    
    dfs(graph, start_vertex, goal_vertex, visited, [], max_path, max_length, 0)

    # Convert max_path from 0-based to 1-based indexing
    max_path: List[int] = [vertex + 1 for vertex in max_path]
    
    return max_path, max_length[0]


# Main function to execute the program
def main() -> int:
    start_time = default_timer()
    # Read teh graph and vertices from the input file
    filename = input("Enter the filename: ")

    try:
        graph, vertices, start_vertex, goal_vertex = read_file(filename)
    except Exception as e:
        print("Error:", e)
        return -1

    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of edges: {len(graph)}")
    print(f"Start vertex: {start_vertex + 1}")
    print(f"Goal vertex: {goal_vertex + 1}")

    euclidean_dist = euclidean_distance(vertices[start_vertex], vertices[goal_vertex])
    print(f"Euclidean distance between {start_vertex + 1} and {goal_vertex + 1}: {euclidean_dist:.4f}")

    # Find the shortest path using Dijkstra's algorithm
    shortest_path, shortest_length = dijkstra(graph, start_vertex, goal_vertex)
    print("Shortest path: ", end="")

    for index in range(len(shortest_path)):
        path = shortest_path[index]
        print(path, end="")
        if index < len(shortest_path) - 1:
            print(" -> ", end="")
    print(f"\nShortest path length: {shortest_length}")

    # Find the longest path using DFS
    longest_path, longest_length = find_longest_path(graph, start_vertex, goal_vertex)
    print("Longest path: ", end="")
    for index in range(len(longest_path)):
        path = longest_path[index]
        print(path, end="")
        if index < len(longest_path) - 1:
            print(" -> ", end="")
    print(f"\nLongest path length: {longest_length}")

    print(f"Time taken for path length: {default_timer() - start_time}" )
    return 0
       
# Run the main function if the script is executed
if __name__ == "__main__":
    main()
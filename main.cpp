#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <memory>

template <typename T1, typename T2>
struct CustomPair {
    T1 first;
    T2 second;

    CustomPair() = default;

    CustomPair(const T1& first_val, const T2& second_val) 
        : first(first_val), second(second_val) {}
};

class PriorityQueue {
public:
    PriorityQueue(int size) : size(0), capacity(size) {
        elements = std::make_unique<CustomPair<int, int>[]>(capacity); // Priority, Item
    }

    bool is_empty() const {
        return size == 0;
    }

    void push(int priority, int item) {
        elements[size++] = {priority, item};
        sift_up(size - 1);
    }

    int pop() {
        if (is_empty()) {
            throw std::runtime_error("Priority queue is empty");
        }
        swap(elements[0], elements[size - 1]);

        int item = elements[size - 1].second;
        size--;
        sift_down(0);
        return item;
    }

private:
    std::unique_ptr<CustomPair<int, int>[]> elements; // Dynamic array for elements
    int size;
    int capacity;

    void sift_up(int index) {
        int parent = (index - 1) / 2;
        if (index > 0 && elements[index].first < elements[parent].first) {
            swap(elements[index], elements[parent]);
            sift_up(parent);
        }
    }

    void sift_down(int index) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        int smallest = index;

        if (left < size && elements[left].first < elements[smallest].first) {
            smallest = left;
        }
        if (right < size && elements[right].first < elements[smallest].first) {
            smallest = right;
        }
        if (smallest != index) {
            swap(elements[smallest], elements[index]);
            sift_down(smallest);
        }
    }

    template <typename T>
    void swap(T& a, T& b) {
        T temp = a;
        a = b;
        b = temp;        
    }
};

class CustomSet {
public:
    CustomSet(int max_size = 100) : size(0), capacity(max_size) {
        items = std::make_unique<int[]>(capacity);
    }

    void add(int item) {
        if (!contains(item)) {
            items[size++] = item;
        }
    }

    void remove(int item) {
        int index = find_index(item);
        if (index != -1) {
            items[index] = items[--size]; // Swap with last item and decrement size
        }
    }

    bool contains(int item) const {
        return find_index(item) != -1;
    }

    int get_size() const {
        return size;
    }

private:
    std::unique_ptr<int[]> items; // Assuming a maximum of 100 items
    int size;
    int capacity;

    int find_index(int item) const {
        for (int i = 0; i < size; ++i) {
            if (items[i] == item) {
                return i;
            }
        }
        return -1; // Item not found
    }
};

double euclidean_distance(const CustomPair<double, double>& v1, const CustomPair<double, double>& v2) {
    return std::sqrt(std::pow(v1.first - v2.first, 2) + std::pow(v1.second - v2.second, 2));
}

void read_file(const std::string& filename, int& start_vertex, int& goal_vertex,
               std::vector<CustomPair<double, double>>& vertices,
               std::vector<std::vector<CustomPair<int, int>>>& graph, int& edge_count) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    int n, m;
    file >> n >> m;
    edge_count = 0;  // Initialize edge counter

    vertices.resize(n);
    graph.resize(n);

    for (int i = 0; i < n; ++i) {
        int vertex;
        double x, y;
        file >> vertex >> x >> y;
        vertices[vertex - 1] = {x, y};
    }

    for (int i = 0; i < m; ++i) {
        int start, end, weight;
        file >> start >> end >> weight;
        graph[start - 1].emplace_back(end - 1, weight);
        edge_count++;
    }

    file >> start_vertex >> goal_vertex;
    start_vertex--;
    goal_vertex--;
}


CustomPair<std::vector<int>, double> dijkstra(const std::vector<std::vector<CustomPair<int, int>>>& graph,
                                              int start_vertex, int goal_vertex) {
    int num_vertices = graph.size();
    double* distances = new double[num_vertices];
    int* parent = new int[num_vertices];
    for (int index = 0; index < num_vertices; ++index) {
        distances[index] = std::numeric_limits<double>::infinity();
        parent[index] = -1;
    }
    distances[start_vertex] = 0;

    PriorityQueue pq(num_vertices);
    pq.push(0, start_vertex);

    while (!pq.is_empty()) {
        int current_vertex = pq.pop();

        if (current_vertex == goal_vertex) {
            break;
        }

        for (int index = 0; index < graph[current_vertex].size(); ++index) {
            CustomPair<int,int> neighbor = graph[current_vertex][index];
            int neighbor_vertex = neighbor.first;
            int weight = neighbor.second;
            double new_distance = distances[current_vertex] + weight;

            if (new_distance < distances[neighbor_vertex]) {
                distances[neighbor_vertex] = new_distance;
                parent[neighbor_vertex] = current_vertex;
                pq.push(new_distance, neighbor_vertex);
            }
        }
    }

    std::vector<int> path;
    int current = goal_vertex;
    while (current != -1) {
        path.push_back(current + 1); // Convert back to 1-based index
        current = parent[current];
    }

    // Manually reverse the path
    std::vector<int> reversed_path;
    for (int i = path.size() - 1; i >= 0; --i) {
        reversed_path.push_back(path[i]);
    }

    delete[] distances;
    delete[] parent;

    return {reversed_path, distances[goal_vertex]};
}

void dfs(const std::vector<std::vector<CustomPair<int, int>>>& graph, int current_vertex, int goal_vertex,
         CustomSet& visited, std::vector<int>& path, std::vector<int>& max_path, int& max_length, int current_length) {
    // Mark the current vertex as visited and add it to the path
    visited.add(current_vertex);
    path.push_back(current_vertex);

    // If we reach the goal, check if this path is the longest we've found
    if (current_vertex == goal_vertex) {
        if (current_length > max_length) {
            max_length = current_length;
            max_path = path; // Store the current path as the longest one found
        }
    } else {
        // Explore each neighbor of the current vertex
        for (const CustomPair<int, int>& neighbor : graph[current_vertex]) {
            int neighbor_vertex = neighbor.first;
            int weight = neighbor.second;

            // Explore the neighbor if it has not been visited yet
            if (!visited.contains(neighbor_vertex)) {
                dfs(graph, neighbor_vertex, goal_vertex, visited, path, max_path, max_length, current_length + weight);
            }
        }
    }

    // Backtrack: Remove the current vertex from the path and mark it as unvisited
    path.pop_back();  // Backtrack the path before returning to the previous recursion state
    visited.remove(current_vertex);  // Ensure we can revisit this vertex in other potential paths
}


std::pair<std::vector<int>, int> find_longest_path(const std::vector<std::vector<CustomPair<int, int>>>& graph,
                                                   int start_vertex, int goal_vertex) {
    CustomSet visited;
    std::vector<int> max_path;
    int max_length = 0;

    std::vector<int> path;
    dfs(graph, start_vertex, goal_vertex, visited, path, max_path, max_length, 0);

    // Convert max_path from 0-based to 1-based indexing
    for (int& vertex : max_path) {
        vertex++;  // Increment each vertex by 1
    }

    return {max_path, max_length};
}

int main() {
    std::string filename;
    std::cout << "Enter the filename: ";
    std::cin >> filename;

    std::vector<std::vector<CustomPair<int, int>>> graph;
    std::vector<CustomPair<double, double>> vertices;
    int start_vertex, goal_vertex, edge_count;
    
    try {
        read_file(filename, start_vertex, goal_vertex, vertices, graph, edge_count);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Number of vertices: " << vertices.size() << std::endl;
    std::cout << "Number of edges: " << edge_count << std::endl;
    std::cout << "Start vertex: " << start_vertex + 1 << std::endl;
    std::cout << "Goal vertex: " << goal_vertex + 1 << std::endl;

    double euclidean_dist = euclidean_distance(vertices[start_vertex], vertices[goal_vertex]);
    std::cout << "Euclidean distance between " << start_vertex + 1 << " and " << goal_vertex + 1 << ": " << euclidean_dist << std::endl;

    auto [shortest_path, shortest_length] = dijkstra(graph, start_vertex, goal_vertex);
    std::cout << "Shortest path: ";
    for (int index = 0; index < shortest_path.size(); index++) {
        int vertex = shortest_path[index];
        std::cout << vertex;
        if (index < shortest_path.size() - 1) {
            std::cout << "->";
        }
    }
    std::cout << "\nShortest length: " << shortest_length << std::endl;

    auto [longest_path, longest_length] = find_longest_path(graph, start_vertex, goal_vertex);
    std::cout << "Longest path: ";
    for (int index = 0; index < longest_path.size(); index++) {
        int vertex = longest_path[index];
        std::cout << vertex;
        if (index < longest_path.size() - 1) 
            std::cout << "->";
    }
    std::cout << "\nLongest length: " << longest_length << std::endl;

    return 0;
}


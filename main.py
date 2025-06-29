import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import time
import threading
from graph import Graph
from algorithms import dijkstra, bellman_ford, floyd_warshall, johnson, prim, kruskal


class GraphAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Algorithm Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        self.graph = None
        self.execution_times = {}
        self.results = {}

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        self.setup_controls(main_frame)
        self.setup_visualization(main_frame)

    def setup_controls(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Graph Configuration", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        ttk.Label(control_frame, text="Vertices:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.vertices_var = tk.StringVar(value="8")
        ttk.Entry(control_frame, textvariable=self.vertices_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(control_frame, text="Density (0-1):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.density_var = tk.StringVar(value="0.3")
        ttk.Entry(control_frame, textvariable=self.density_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(control_frame, text="Max Weight:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.weight_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.weight_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        self.directed_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Directed Graph", variable=self.directed_var).grid(row=3, column=0,
                                                                                               columnspan=2,
                                                                                               sticky=tk.W, pady=5)

        ttk.Button(control_frame, text="Generate Graph", command=self.generate_graph).grid(row=4, column=0,
                                                                                           columnspan=2, pady=10)

        ttk.Label(control_frame, text="Shortest Path Algorithms:").grid(row=5, column=0, columnspan=2, sticky=tk.W,
                                                                        pady=(20, 5))

        self.dijkstra_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Dijkstra", variable=self.dijkstra_var).grid(row=6, column=0, sticky=tk.W)

        self.bellman_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Bellman-Ford", variable=self.bellman_var).grid(row=7, column=0,
                                                                                            sticky=tk.W)

        self.floyd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Floyd-Warshall", variable=self.floyd_var).grid(row=8, column=0,
                                                                                            sticky=tk.W)

        self.johnson_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Johnson", variable=self.johnson_var).grid(row=9, column=0, sticky=tk.W)

        ttk.Label(control_frame, text="Start Node:").grid(row=10, column=0, sticky=tk.W, pady=(10, 2))
        self.start_var = tk.StringVar(value="1")
        ttk.Entry(control_frame, textvariable=self.start_var, width=10).grid(row=10, column=1, sticky=tk.W,
                                                                             pady=(10, 2))

        ttk.Label(control_frame, text="End Node:").grid(row=11, column=0, sticky=tk.W, pady=2)
        self.end_var = tk.StringVar(value="2")
        ttk.Entry(control_frame, textvariable=self.end_var, width=10).grid(row=11, column=1, sticky=tk.W, pady=2)

        ttk.Button(control_frame, text="Run Path Algorithms", command=self.run_path_algorithms).grid(row=12, column=0,
                                                                                                     columnspan=2,
                                                                                                     pady=10)

        ttk.Label(control_frame, text="MST Algorithms:").grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))

        self.prim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Prim", variable=self.prim_var).grid(row=14, column=0, sticky=tk.W)

        self.kruskal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Kruskal", variable=self.kruskal_var).grid(row=15, column=0, sticky=tk.W)

        ttk.Button(control_frame, text="Run MST Algorithms", command=self.run_mst_algorithms).grid(row=16, column=0,
                                                                                                   columnspan=2,
                                                                                                   pady=10)

        self.results_text = tk.Text(control_frame, height=8, width=30)
        self.results_text.grid(row=17, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=17, column=2, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)

    def setup_visualization(self, parent):
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Graph")

        self.perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.perf_frame, text="Performance")

        self.setup_graph_plot()
        self.setup_performance_plot()

    def setup_graph_plot(self):
        self.fig_graph, self.ax_graph = plt.subplots(figsize=(8, 6))
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, self.graph_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_performance_plot(self):
        self.fig_perf, self.ax_perf = plt.subplots(figsize=(8, 6))
        self.canvas_perf = FigureCanvasTkAgg(self.fig_perf, self.perf_frame)
        self.canvas_perf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_graph(self):
        try:
            vertices = int(self.vertices_var.get())
            density = float(self.density_var.get())
            max_weight = int(self.weight_var.get())
            directed = self.directed_var.get()

            if vertices < 2 or vertices > 50:
                messagebox.showerror("Error", "Vertices must be between 2 and 50")
                return

            if density < 0 or density > 1:
                messagebox.showerror("Error", "Density must be between 0 and 1")
                return

            self.graph = Graph(vertices, density, max_weight, directed)
            self.visualize_graph()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Graph generated: {vertices} vertices, density={density:.2f}\n")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")

    def visualize_graph(self, highlight_path=None, highlight_mst=None):
        if not self.graph:
            return

        self.ax_graph.clear()

        G = nx.DiGraph() if self.graph.directed else nx.Graph()
        G.add_nodes_from(self.graph.vertices)

        for u, v, weight in self.graph.edges:
            G.add_edge(u, v, weight=weight)

        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        for node in G.nodes():
            if highlight_path and len(highlight_path) > 1:
                if node == highlight_path[0]:
                    node_colors.append('lightgreen')
                elif node == highlight_path[-1]:
                    node_colors.append('lightcoral')
                elif node in highlight_path:
                    node_colors.append('yellow')
                else:
                    node_colors.append('lightblue')
            else:
                node_colors.append('lightblue')

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=self.ax_graph)
        nx.draw_networkx_labels(G, pos, ax=self.ax_graph)

        edge_colors = []
        for u, v in G.edges():
            if highlight_path and len(highlight_path) > 1:
                path_edges = [(highlight_path[i], highlight_path[i + 1]) for i in range(len(highlight_path) - 1)]
                if (u, v) in path_edges or (v, u) in path_edges:
                    edge_colors.append('red')
                else:
                    edge_colors.append('gray')
            elif highlight_mst:
                mst_edges = [(edge[0], edge[1]) for edge in highlight_mst]
                if (u, v) in mst_edges or (v, u) in mst_edges:
                    edge_colors.append('red')
                else:
                    edge_colors.append('gray')
            else:
                edge_colors.append('gray')

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=self.ax_graph)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=self.ax_graph)

        self.ax_graph.set_title("Graph Visualization")
        self.canvas_graph.draw()

    def run_algorithm_with_timing(self, algorithm, *args):
        start_time = time.perf_counter()
        result = algorithm(*args)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        return result, execution_time

    def run_path_algorithms(self):
        if not self.graph:
            messagebox.showerror("Error", "Please generate a graph first")
            return

        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())

            if start < 1 or start > len(self.graph.vertices) or end < 1 or end > len(self.graph.vertices):
                messagebox.showerror("Error", f"Start and end nodes must be between 1 and {len(self.graph.vertices)}")
                return

        except ValueError:
            messagebox.showerror("Error", "Please enter valid node numbers")
            return

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Running path algorithms from {start} to {end}...\n\n")
        self.root.update()

        algorithms_to_run = []
        if self.dijkstra_var.get():
            algorithms_to_run.append(("Dijkstra", dijkstra))
        if self.bellman_var.get():
            algorithms_to_run.append(("Bellman-Ford", bellman_ford))
        if self.floyd_var.get():
            algorithms_to_run.append(("Floyd-Warshall", floyd_warshall))
        if self.johnson_var.get():
            algorithms_to_run.append(("Johnson", johnson))

        results = {}
        times = {}

        for name, algorithm in algorithms_to_run:
            try:
                result, exec_time = self.run_algorithm_with_timing(algorithm, self.graph, start, end)
                results[name] = result
                times[name] = exec_time

                self.results_text.insert(tk.END, f"{name}:\n")
                self.results_text.insert(tk.END, f"  Path: {result if result else 'No path found'}\n")
                self.results_text.insert(tk.END, f"  Time: {exec_time:.3f} ms\n\n")
                self.root.update()

            except Exception as e:
                self.results_text.insert(tk.END, f"{name}: Error - {str(e)}\n\n")

        for name, path in results.items():
            if path:
                self.visualize_graph(highlight_path=path)
                break

        if times:
            self.update_performance_chart(times, "Shortest Path Algorithms")

    def run_mst_algorithms(self):
        if not self.graph:
            messagebox.showerror("Error", "Please generate a graph first")
            return

        if self.graph.directed:
            messagebox.showwarning("Warning",
                                   "MST algorithms work on undirected graphs. Results may not be meaningful.")

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running MST algorithms...\n\n")
        self.root.update()

        algorithms_to_run = []
        if self.prim_var.get():
            algorithms_to_run.append(("Prim", prim))
        if self.kruskal_var.get():
            algorithms_to_run.append(("Kruskal", kruskal))

        results = {}
        times = {}

        for name, algorithm in algorithms_to_run:
            try:
                result, exec_time = self.run_algorithm_with_timing(algorithm, self.graph)
                results[name] = result
                times[name] = exec_time

                total_weight = sum(edge[2] for edge in result) if result else 0

                self.results_text.insert(tk.END, f"{name}:\n")
                self.results_text.insert(tk.END, f"  Edges: {len(result) if result else 0}\n")
                self.results_text.insert(tk.END, f"  Total weight: {total_weight}\n")
                self.results_text.insert(tk.END, f"  Time: {exec_time:.3f} ms\n\n")
                self.root.update()

            except Exception as e:
                self.results_text.insert(tk.END, f"{name}: Error - {str(e)}\n\n")

        for name, mst in results.items():
            if mst:
                self.visualize_graph(highlight_mst=mst)
                break

        if times:
            self.update_performance_chart(times, "MST Algorithms")

    def update_performance_chart(self, times, title):
        self.ax_perf.clear()

        algorithms = list(times.keys())
        execution_times = list(times.values())

        bars = self.ax_perf.bar(algorithms, execution_times, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])

        self.ax_perf.set_ylabel('Execution Time (ms)')
        self.ax_perf.set_title(f'{title} - Performance Comparison')
        self.ax_perf.tick_params(axis='x', rotation=45)

        for bar, time in zip(bars, execution_times):
            height = bar.get_height()
            self.ax_perf.text(bar.get_x() + bar.get_width() / 2., height,
                              f'{time:.3f}ms', ha='center', va='bottom')

        self.fig_perf.tight_layout()
        self.canvas_perf.draw()


def main():
    root = tk.Tk()
    app = GraphAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

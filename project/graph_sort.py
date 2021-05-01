from collections import OrderedDict, deque
import datetime


class DAG(object):
    """ Directed acyclic graph implementation. """

    def __init__(self):
        """ Construct a new DAG with no nodes or edges. """
        self.node_depth = []
        self.graph = OrderedDict()

    def add_node(self, node: tuple):
        """ Add a node if it does not exist yet, or error out. """
        if node in self.graph:
            raise KeyError('node %s already exists' % node.index)
        self.graph[node] = set()

    def add_edge(self, start_node: tuple, end_node: tuple):
        """ Add an edge (dependency) between the specified nodes. """

        if start_node not in self.graph or end_node not in self.graph:
            raise KeyError("Node is not existed in the graph.")

        self.graph[start_node].add(end_node)

    # Sort children node by time in ascending order
    def sort_children(self):
        for key in self.graph:
            self.graph[key] = sorted(self.graph[key], key=lambda item: item[1])

    def get_graph_dict(self):
        return self.graph

    def topological_sort(self):
        """ Returns a topological ordering of the DAG.
        Raises an error if this is not possible (graph is not valid).
        """

        in_degree = {}
        for u in self.graph:
            in_degree[u] = 0

        for u in self.graph:
            for v in self.graph[u]:
                in_degree[v] += 1

        queue = deque()
        for u in in_degree:
            if in_degree[u] == 0:
                queue.appendleft(u)

        l = []
        while queue:
            u = queue.pop()
            l.append(u)
            for v in self.graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)

        if len(l) == len(self.graph):
            return l
        else:
            raise ValueError('graph is not acyclic')


def create_graph(items):
    graph = DAG()
    root_time = ""
    idStr_idx = {}

    for idx, item in enumerate(items):
        if idx == 0:
            root_time = item["created_at"]
        commit_time = item["created_at"]
        idStr_idx.update({item["id_str"]: idx})  # Assuming every tweet is unique
        graph.add_node((idx, calc_time_diff(root_time, commit_time)))

    keys = list(graph.graph.copy())

    for idx, item in enumerate(items):
        parent_idx = idStr_idx.get(str(item["in_reply_to_status_id_str"]))
        if parent_idx is not None:
            graph.add_edge(keys[parent_idx], keys[idx])

    graph.sort_children()

    return graph


def calc_time_diff(start_time: str, end_time: str):
    start_time_formatted = datetime.datetime.strptime(start_time, "%a %b %d %H:%M:%S %z %Y")
    end_time_formatted = datetime.datetime.strptime(end_time, "%a %b %d %H:%M:%S %z %Y")
    return (end_time_formatted - start_time_formatted).total_seconds()

# import networkx as nx

class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """
    
    def __init__(self):
        self.node_ids = []
        self.successors = {}
        self.predecessors = {}
        
    def add_edge(self, u: str, v: str):
        if u in self.node_ids and v in self.node_ids:
            self.successors[u].append(v)
            self.predecessors[v].append(u)

    def add_node(self, node: str):
        if not(node in self.node_ids):
            self.node_ids.append(node)
            self.successors[node] = []
            self.predecessors[node] = []

    def get_successors(self, node: str):
        return self.successors[node]

    def get_predecessors(self, node: str):
        return self.predecessors[node]

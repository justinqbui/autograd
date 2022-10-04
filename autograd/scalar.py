class Scalar:
    def __init__(self, value, children=(), operation=None):
        self.value = value
        self.grad = 0.
        self.children = children
        self.operation = operation

    def __repr__(self):
        return f"Scalar({self.value}), with grad: {self.grad} and operation: {self.operation}"

    def __add__(self, other):
        return Scalar(self.value + other.value, children=[self, other], operation="+")
    
    def __mul__(self, other):
        return Scalar(self.value * other.value, children=[self, other], operation="*")

    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * -1
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def backward(self):
        visited = set()
        topological = []
        self._topologicalSort(visited, topological)

        # we start with the gradient of the final node, which is 1
        topological[-1].grad = 1.

        # initialize the gradient of the root node
        for node in reversed(topological):
            if node.operation == "+":
                node.children[0].grad += 1.0 * node.grad
                node.children[1].grad += 1.0 * node.grad
            elif node.operation == "*":
                node.children[0].grad += (node.grad * node.children[1].value)
                node.children[1].grad += (node.grad * node.children[0].value)

    # helper function for building the topological sort for backprop
    def _topologicalSort(self, visited, topological):
        if self in visited:
            return
        visited.add(self)
        for child in self.children:
            child._topologicalSort(visited, topological)
        topological.append(self)
        
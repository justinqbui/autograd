class Scalar:
    def __init__(self, 
    value: float, 
    children: list=[], 
    operation: str=None):
        self.value = value
        self.grad = 0.
        self.children = children
        self.operation = operation

    def __repr__(self) -> str:
        return f"{self.value:.2f}"
        # return f"Scalar({self.value}), with grad: {self.grad} and operation: {self.operation}"

    def __add__(self, other: 'Scalar'):
        return Scalar(self.value + other.value, children=[self, other], operation="+")
    
    def __mul__(self, other: 'Scalar'):
        return Scalar(self.value * other.value, children=[self, other], operation="*")

    def __sub__(self, other: 'Scalar') -> 'Scalar':
        return self + (-other)
    
    def __neg__(self) -> 'Scalar':
        return self * -1
    
    def __rmul__(self, other: 'Scalar') -> 'Scalar':
        return self * other
    
    def __radd__(self, other: 'Scalar') -> 'Scalar':
        return self + other
    
    def backward(self) -> None:
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
            elif node.operation == "relu":
                if node.value > 0:
                    node.children[0].grad += node.grad * (node.value > 0)

    # helper function for building the topological sort for backprop
    def _topologicalSort(self, visited: set, topological: list) -> None:
        if self in visited:
            return
        visited.add(self)
        for child in self.children:
            child._topologicalSort(visited, topological)
        topological.append(self)
    
    def relu(self):
        return Scalar(max(0, self.value), children=[self], operation="relu")
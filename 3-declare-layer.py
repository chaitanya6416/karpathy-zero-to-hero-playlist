
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

# x1 = Value(0.5)
# x2 = Value(1.3)

# w1 = Value(0.5)
# w2 = Value(0.5)
# b = Value(0.5)
# parameters = [w1, w2, b]

# o = ((x1*w1) + (x2*w2) + (b)).relu()
# def make_forward_pass_and_get_result():
#     return x1*w1 + x2*w2 + b

# desired_o = 0.2
# itr = 0
# learning_rate = 0.001

# for _ in range(10):
#     # 1. calculate loss 
#     pred_o = make_forward_pass_and_get_result()
#     loss = (pred_o - desired_o)**2
#     print(f"iteration {itr}; {loss=}"); itr+=1
    
#     # 2. calculate grads to tweak weights in the next step
#     for param in parameters:
#         param.grad = 0.0
#     o.backward()
    
#     # 3. update the weights of parameters along the -gradient
#     for param in parameters:
#         param.data += -1 * learning_rate * param.grad

#     learning_rate -= (learning_rate/10)

# print(parameters)

from random import random
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random()) for _ in range(nin)]
        self.b = Value(random())
    def __call__(self, x):
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = out.relu()
        return out
    def __repr__(self):
        return f"{self.w=} {self.b=}"

# n1 = Neuron(2); print(n1)
# x = [0.2, 0.5]
# o = n1(x)
# print(o)

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)] # n_outputs == number_of_neurons in that layer 
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out

l1 = Layer(2, 5)
x = [0.2, 0.5]
o = l1(x)
print(o)

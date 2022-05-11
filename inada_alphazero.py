from inada_framework import Model, optimizers, cuda, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf



class PolicyValueNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.p = dzl.Affine(action_size)
        self.v = dzl.Affine(1)

    def forward(self, x):
        policy = dzf.softmax(self.p(x))
        value = dzf.tanh(self.v(x))
        return policy, value



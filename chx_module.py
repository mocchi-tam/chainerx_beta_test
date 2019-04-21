import numpy as np
import chainerx as chx

# chx Link
class Linear(object):
	def __init__(self, n_in, n_out):
		W = np.random.randn(n_in, n_out).astype(np.float32)
		W /= np.sqrt(n_in)
		self.W = chx.array(W)
		self.b = chx.zeros((n_out,), dtype=chx.float32)
		
	def __call__(self, x):
		x = x.reshape(x.shape[:2])
		return x.dot(self.W) + self.b
	
	@property
	def params(self):
		return self.W, self.b

# chx Optimizer
class SGD(object):
	def __init__(self, lr=0.01):
		super(SGD, self).__init__()
		self.lr = lr
	
	def setup(self, model):
		self.layers = model.layers
		# require grad
		for layer in self.layers:
			for param in layer.params:
				param.require_grad()
				
	def update(self):
		for layer in self.layers:
			for param in layer.params:
				p = param.as_grad_stopped()
				p -= self.lr * param.grad.as_grad_stopped()
				param.cleargrad()
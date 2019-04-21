import argparse
import time
import numpy as np
import chainer 
import chainer.links as L
import chinaer.functions as F
from chainer.dataset import convert
import chainerx as chx
# my module
from chx_module import Linear, SGD

# chainer: MLP
class MLP_chain(chainer.Chain):
	def __init__(self, n_units=1000, n_out=10):
		super(MLP_chain, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(784, n_units)
			self.l2 = L.Linear(n_units, n_units)
			self.l3 = L.Linear(n_units, n_out)
	
	def forward(self, x):
		h = F.relu(self.l1(x))
		h = F.relu(self.l2(h))
		return self.l3(h)

# chainerx: MLP
class MLP(object):
	def __init__(self, n_units=1000, n_out=10):
		self.l1 = Linear(784, n_units)
		self.l2 = Linear(n_units, n_units)
		self.l3 = Linear(n_units, n_out)
	
	def forward(self, x):
		h = chx.relu(self.l1(x))
		h = chx.relu(self.l2(h))
		return self.l3(h)
	
	@property
	def layers(self):
		return (self.l1, self.l2, self.l3)

# chainerx: softmax cross entropy
def compute_loss(y, t):
	score = chx.log_softmax(y, axis=1)
	mask = (t[:, chx.newaxis]==chx.arange(10, dtype=t.dtype)).astype(score.dtype)
	return -(score * mask).sum() * (1 / y.shape[0])

# main
def main():
	parser = argparse.ArgumentParser(description='Compare chainer vs chainerx')
	parser.add_argument('--batchsize', '-b', type=int, default=100)
	parser.add_argument('--epoch', '-e', type=int, default=10)
	parser.add_argument('--gpu', '-g', type=int, default=0, choices=[-1, 0, 1, 2, 3])
	parser.add_argument('--chxon', '-c', type=int, default=1)
	args = parser.parse_args()
	
	# setup
	start = time.time()
	chx.available = True if args.chxon == 1 else False
	batch_size = args.batchsize
	
	# get MNIST
	train, test = chainer.datasets.get_mnist()
	
	if chx_available == True:
		device_name = 'cuda:{}'.format(args.gpu)
		# data
		with chx.using_device(device_name):
			train_images, train_labels = map(lamda d:chx.asarray(d), train._datasets)
			test_images, test_labels = map(lamda d:chx.asarray(d), test._datasets)
		# model
		chx.set_default_device(device_name)
		model = MLP(n_units=1000, n_out=10)
		optimizer = SGD(lr=0.01)
	else:
		device_name = args.gpu
		# data
		train_iter = chainer.iterators.SerialIterator(train, batch_size)
		test_iter = chainer.iterators.SerialIterator(train, batch_size, repeat=False, shuffle=False)
		# model
		model = MLP_chain(n_units=1000, n_out=10)
		model.to_gpu()
		chainer.cuda.get_device_from_id(device_name).use()
		optimizer = chainer.optimizers.SGD(lr=0.01)
	
	optimizer.setup(model)
	
	N_train, N_test = len(train), len(test)
	all_indices_np = np.arange(N_train, dtype=np.int64) # for chainerx
	epoch = 0
	
	while epoch <= args.epoch:
		epoch += 1
		if chx_available == True:
			np.random.shuffle(all_indices_np)
			all_indices = chx.array(all_indices_np)
		
		for i in range(0, N_train, batch_size):
			# time 1
			if chx_available == True:
				indices = all_indices[i: i + batch_size]
				x = train_images.take(indices, axis=0)
				t = train_labels.take(indices, axis=0)
			else:
				batch = train_iter.next()
				x, t = convert.concat_examples(batch, device=device_name)
			
			y = model.forward(x) # time 2
			
			# time 3
			if chx_available == True:
				loss = compute_loss(y, t)
			else:
				loss = F.softmax_cross_entropy(y, t)
				model.cleargrads()
			
			loss.backward() # time 4
			optimizer.update() # time 5
		
		if chx_available == True:
			with chx.no_backprop_mode():
				total_loss = chx.array(0, dtype=chx.float32)
				num_correct = chx.array(0, dtype=chx.int64)
				for i in range(0, N_test, batch_size):
					x = test_images[i:min(i + batch_size, N_test)]
					x = test_labels[i:min(i + batch_size, N_test)]
					
					y = model.forward(x)
					total_loss += compute_loss(y, t) * len(t)
					num_correct += (y.argmax(axis=1).astype(t.dtype) == t).astype(chx.int32).sum()
		else:
			test_iter.reset()
			with chainer.using_config('enable_backprop', False):
				total_loss = 0
				num_correct = 0
				for batch in test_iter:
					x, t = convert.concat_examples(batch, device=device_name)
					
					y = model.forward(x)
					total_loss += float(F.softmax_cross_entropy(y, t).array) * len(t)
					num_correct += float(F.accuracy(y, t).array) * len(t)
			
			mean_loss = float(total_loss) / N_test
			accuracy = int(num_correct) / N_test
			elapsed_time = time.time() - start
			print('epoch {} ... loss={}, accuracy, elapsed_time={}'.format(
						epoch, mean_loss, accuracy, elapsed_time))

if __name__ = '__main__':
	main()
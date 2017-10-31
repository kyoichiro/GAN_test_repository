#coding:utf-8

import numpy as np
import chainer
from chainer import Function, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata

import matplotlib.pyplot as plt
import gzip
import sys

N = 60000 

class generator(Chain):
	def __init__(self):
		super(generator, self).__init__(
			gl1=L.Linear(4,3),
			gl2=L.Linear(3,4),
		)

	def __call__(self, x, m):
		fv = self.fwd(x)
		loss = log(1-Discriminator.fwd(fv))/m
		return loss

	def fwd(self, x):
		h1=F.sigmoid(self.gl1(x)) 
		h2=F.sigmoid(self.gl2(h1))
		return h2


class Discriminator(Chain):
	def __init__(self):
		super(Discriminator, self).__init__(
			dl1=L.Linear(4,3),
			dl2=L.Linear(3,1),
		)

	def __call__(self, x, m):
		fv = self.fwd(x)
		loss = log(fv)+log(1-Discriminator.fwd(generator.fwd(x)))/m
		return  loss

	def fwd(self, x):
		h1=F.sigmoid(self.dl1(x))
		h2=F.sigmoid(self.dl2(h1))
		return h2

if __name__ == "__main__":

	#setup models 
	generator = generator()
	Discriminator = Discriminator()
	opt_gene = optimizers.SGD()
	opt_dis = optimizers.SGD()
	opt_gene.setup(generator)
	opt_dis.setup(Discriminator)

	#reload(sys)
	#sys.setdefaultencoding('utf8')

	print("start:import MNIST")

	mnist = fetch_mldata('MNIST original', data_home=".")
	#with gzip.open('/Users/admin/Desktop/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
		#mnist = f.read()

	print("end:import MNIST")

	print(type(mnist.data))
	print(len(mnist.data))

	#mnist.data : 70,000件の28x28=784次元ベクトルデータ
	mnist = mnist.data.astype(np.float32)
	mnist /= 255  # 正規化
	
	#X = mnist.data
	X = mnist
	X = X.reshape(70000,28,28)

	#print(type(X))
	#print(len(X))

	plt.figure(figsize = (28, 28))
	print(len(X[0])) 
	
	cnt=0
	'''
	for i in np.random.permutation(N)[:100]:
		cnt+=1
		plt.subplot(10,10,cnt)
		X[i] = X[i][::-1] #inverse of number
		plt.xlim(0,27)
		plt.ylim(0,27)
		plt.imshow(X[i])
		plt.pcolor(X[i])
		plt.gray()
	plt.show()
	'''
	print("End program")











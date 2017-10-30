import numpy as np
import chainer
from chainer import Function, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata
import matplotlib as plt
import gzip
import sys

class Generater(Chain):
	def __init__(self):
		super(Generater, self).__init__(
			gl1=L.Linear(4,3),
			gl2=L.Linear(3,4),
		)

	def __call__(self, x, y):
		fv = self.fwd(x)
		loss = F.mean_squared_error(fv, y)
		return loss

	def fwd(self, x):
		h1=F.sigmoid(self.gl1(x)) 
		h2=F.sigmoid(self.gl2(h1))
		return h2


class Discriminater(Chain):
	def __init__(self):
		super(Discriminater, self).__init__(
			dl1=L.Linear(4,3),
			dl2=L.Linear(3,1),
		)

	def __call__(self, x, y):
		fv = self.fwd(x)
		loss = F.mean_squared_error(fv, y)
		return  loss

	def fwd(self, x):
		h1=F.sigmoid(self.dl1(x))
		h2=F.sigmoid(self.dl2(h1))


if __name__ == "__main__":

	#setup models 
	Generater = Generater()
	Discriminater = Discriminater()
	opt_gene = optimizers.SGD()
	opt_dis = optimizers.SGD()
	opt_gene.setup(Generater)
	opt_dis.setup(Discriminater)

	#reload(sys)
	#sys.setdefaultencoding('utf8')

	#mnist = fetch_mldata('MNIST original', data_home="./MNIST")
	with gzip.open('/Users/admin/Desktop/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
		mnist = f.read()

	print(type(mnist))
	print(len(mnist))

	#mnist = str(mnist)

	# mnist.data : 70,000件の28x28=784次元ベクトルデータ
	#mnist = mnist.data.astype(np.float32)
	#mnist.data /= 255  # 正規化
	mnist = np.array(mnist)
	mnist = mnist.astype(np.float32)
	mnist /= 255

	#X = mnist.data
	X = mnist

	#print(type(X))
	#print(len(X))

	plt.figure(figsize = (15, 15))   
	
	for i in range(10): 
	    plt.subplot(10,10,i)
	    X = X.reshape(28,28)
	    X = X[::-1]
	    plt.xlim(0,27)
	    plt.ylim(0,27)
	    plt.imshow(X[i])
	    plt.gray()
	plt.show()











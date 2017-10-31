#coding:utf-8

import numpy as np
import chainer
from chainer import Function, Variable
from chainer import cuda, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata

import matplotlib.pyplot as plt
import gzip
import sys

N = 60000 
input_units = 784
n_units = 400
n2_units = 100
output_units = 10
n_epoch = 10
batchsize = 100

class Generator(Chain):
	def __init__(self):
		super(Generator, self).__init__(
			gl1=L.Linear(input_units, n_units),
			gl2=L.Linear(n_units, n2_units),
			gl3=L.Linear(n2_units, output_units),
		)

	def __call__(self, x, m):
		fv = self.fwd(x)
		loss = log(1-Discriminator.fwd(fv))/m
		return loss

	def fwd(self, x):
		h1=F.dropout(F.relu(self.gl1(x))) 
		h2=F.dropout(F.relu(self.gl2(h1)))
		h3=F.dropout(F.relu(self.gl3(h2)))
		return h3


class Discriminator(Chain):
	def __init__(self):
		super(Discriminator, self).__init__(
			dl1=L.Linear(input_units, n_units),
			dl2=L.Linear(n_units, n2_units),
			dl3=L.Linear(n2_units, output_units),
		)

	def __call__(self, x, y):
		fv = self.fwd(x)
		#loss = log(fv)+log(1-Discriminator.fwd(Generator.fwd(x)))/m
		loss = F.softmax_cross_entropy(fv, y)
		acc = F.accuracy(fv, y)
		return  loss,acc

	def fwd(self, x):
		h1=F.dropout(F.relu(self.dl1(x))) 
		h2=F.dropout(F.relu(self.dl2(h1)))
		h3=F.dropout(F.relu(self.dl3(h2)))
		return h3


if __name__ == "__main__":

	#setup models 
	Generator = Generator()
	Discriminator = Discriminator()
	opt_gene = optimizers.SGD()
	opt_dis = optimizers.SGD()
	opt_gene.setup(Generator)
	opt_dis.setup(Discriminator)

	train_loss = []
	train_acc  = []
	test_loss = []
	test_acc  = []

	dl1_W = []
	dl2_W = []
	dl3_W = []

	#reload(sys)
	#sys.setdefaultencoding('utf8')

	print("start:import MNIST")

	mnist = fetch_mldata('MNIST original', data_home=".")
	#with gzip.open('/Users/admin/Desktop/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
		#mnist = f.read()

	print("end:import MNIST")

	#mnist.data : 70,000件の28x28=784次元ベクトルデータ
	mnist.data = mnist.data.astype(np.float32)
	mnist.data /= 255  # 正規化
	mnist.target = mnist.target.astype(np.int32)
	
	x_train, x_test = np.split(mnist.data,   [N])
	y_train, y_test = np.split(mnist.target, [N])
	N_test = y_test.size

	#print(type(X))
	#print(len(X))

	# Learning loop
	for epoch in range(1, n_epoch+1):
		print("epoch: "+ str(epoch))

		# training
		# N個の順番をランダムに並び替える
		perm = np.random.permutation(N)
		sum_accuracy = 0
		sum_loss = 0
		# 0〜Nまでのデータをバッチサイズごとに使って学習
		for i in range(0, N, batchsize):
			x_batch = x_train[perm[i:i+batchsize]]
			y_batch = y_train[perm[i:i+batchsize]]
			x_batch, y_batch = Variable(x_batch), Variable(y_batch)
			# 勾配を初期化
			#Generator.zerograds()
			Discriminator.zerograds()
			# 順伝播させて誤差と精度を算出
			loss, acc = Discriminator(x_batch, y_batch)
			# 誤差逆伝播で勾配を計算
			loss.backward()
			opt_dis.update()
			#opt_gene.update()

			train_loss.append(loss.data)
			train_acc.append(acc.data)
			sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
			sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

		# 訓練データの誤差と、正解精度を表示
		print ("train mean loss={}, accuracy={}".format(sum_loss / N, sum_accuracy / N))

		# evaluation
		# テストデータで誤差と、正解精度を算出し汎化性能を確認
		sum_accuracy = 0
		sum_loss     = 0
		for i in range(0, N_test, batchsize):
			x_batch = x_test[i:i+batchsize]
			y_batch = y_test[i:i+batchsize]

			# 順伝播させて誤差と精度を算出
			loss, acc = Discriminator(x_batch, y_batch)

			test_loss.append(loss.data)
			test_acc.append(acc.data)
			sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
			sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

		# テストデータでの誤差と、正解精度を表示
		print ("test  mean loss={}, accuracy={}".format(sum_loss / N_test, sum_accuracy / N_test))

		# 学習したパラメーターを保存
		dl1_W.append(Discriminator.dl1.W)
		dl2_W.append(Discriminator.dl2.W)
		dl3_W.append(Discriminator.dl3.W)


	#plt.figure(figsize = (28, 28))
	
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

	# 精度と誤差をグラフ描画
	plt.figure(figsize=(8,6))
	plt.plot(range(len(train_acc)), train_acc)
	plt.plot(range(len(test_acc)), test_acc)
	plt.legend(["train_acc","test_acc"],loc=4)
	plt.title("Accuracy of digit recognition.")
	plt.plot()
	plt.show()

	print("End program")











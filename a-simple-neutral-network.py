import numpy as np
#sigmoid激活函数，定义标志位deriv，默认为False不去计算导数前向传播，deriv=True时误差反向传播
def sigmoid(x,deriv = False):    #注意此处传入x是np.array
	if (deriv == True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1]])
#print(x.shape)
y = np.array([[0],[1],[1],[0],[0]])
#print(y.shape)

np.random.seed(1)
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1
#print(w0)

for j in range(60000):
	l0 = x                         #定义l0层,输入特征
	l1 = sigmoid(np.dot(l0,w0))    #定义l1层,用sigmoid激活
	l2 = sigmoid(np.dot(l1,w1))    #定义l2层,用sigmoid激活
	l2_error = y - l2              #真实值与预测值的差异
	if (j%10000) == 0:
		print('Error:'+str(np.mean(np.abs(l2_error))))

	l2_delta = l2_error * sigmoid(l2,deriv=True)    #l2_error作为权重项，错的越多修正越大
	l1_error = l2_delta.dot(w1.T)    #.dot矩阵乘法,l1_error为5行4列
	l1_delta = l1_error * sigmoid(l1,deriv=True)
	#更新权重
	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)
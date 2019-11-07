import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#s=np.random.randint(500)
#print(s)
#np.random.seed(s)
np.random.seed(4)
def rnd_norm():
    return np.random.normal()

#d)
def softmax(f):
    e_f = np.exp(f)
    return e_f/np.sum(e_f, axis=1, keepdims=True)

P_0 = pd.read_hdf('populationen.hdf5', key='P_0')
P_1 = pd.read_hdf('populationen.hdf5', key='P_1')
P_x = np.append(P_0.x, P_1.x)
P_y = np.append(P_0.y, P_1.y)
label = np.append(np.zeros(len(P_0)),np.ones(len(P_1)))
P = np.vstack([P_x, P_y, label])
Pt = P.T[:, :2]
rate = 0.5
epochs = 100
W = np.zeros((2,2))
b = np.zeros(2)
for i in range(len(b)):
    b[i] = rnd_norm()
    W[i][i] = rnd_norm()
    W[~i][i] = rnd_norm()

for i in range(epochs):
    f_i = Pt @ W + b
    smf_i = softmax(f_i)
    smf_i[range(len(P_x)), [int(d) for d in P.T[:, 2]]] -=1
    smf_i /= len(P_x)
    dW = Pt.T @ smf_i
    db = np.array([np.sum(smf_i[:, 0]), np.sum(smf_i[:, 1])])
    W -= rate * dW
    b -= rate * db

print('W: ' ,W)
print('b: ', b)

#e)

def Gerade(X, W, b):
    return (b[1]-b[0]+X*(W[1][0]-W[0][0]))/(W[0][1]-W[1][1])

x = np.linspace(np.min(P_x),np.max(P_x),10000)
plt.plot(P_0.x, P_0.y, 'r.', alpha=0.7, markersize=0.7, label='P_0')
plt.plot(P_1.x, P_1.y, 'b.', alpha=0.7, markersize=0.7, label='P1')
plt.plot(x, Gerade(x, W, b), 'k', label='Gerade')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-10,15)
plt.savefig('build/Trennung.pdf')

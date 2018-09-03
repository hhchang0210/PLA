import matplotlib.pyplot as plt
import numpy as np


#網路上找的dataset 可以線性分割
'''
dataset = np.array([
((1, -0.4, 0.3), -1),
((1, -0.3, -0.1), -1),
((1, -0.2, 0.4), -1),
((1, -0.1, 0.1), -1),
((1, 0.9, -0.5), 1),
((1, 0.7, -0.9), 1),
((1, 0.8, 0.2), 1),
((1, 0.4, -0.6), 1)])
'''

X1 = np.array([-0.62231486, -0.96251306,  0.42269922, -1.452746  , -0.66915783,
               -0.35716016,  0.49505163, -1.8117848 ,  0.53376487, -1.86923838,
                0.71434306, -0.4055084 ,  0.82887254,  0.81221287,  1.44280951,
               -0.45599278, -1.16715888,  1.08913131, -1.61470741,  1.61113001,
               -1.4532688 ,  1.04872588, -1.52312195, -1.62831727, -0.25191539])

X2 = np.array([-1.67427011, -1.81046748,  1.20384694, -0.41572751,  0.66851908,
               -1.75435288, -1.57532207, -1.22329618, -0.84375819,  0.52873296,
               -1.10837773,  0.04612922,  0.67696196,  0.84618152, -0.77362548,
                0.99153072,  1.7896494 , -0.38343121, -0.21337742,  0.64754817,
                0.36719101,  0.23132427,  1.07029963,  1.62919909, -1.53920827])

Y = np.array([  1.,  -1.,  -1.,  -1.,  -1.,
                1.,   1.,  -1.,   1.,  -1.,
                1.,  -1.,   1.,   1.,   1.,
               -1.,  -1.,   1.,  -1.,   1.,
               -1.,   1.,  -1.,  -1.,   1.])

L = []
for i in range(len(X1)):
    L.append( ((1, X1[i], X2[i]), Y[i]) )
print(L)

dataset = np.array(L)



def plot_data():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X1[Y >= 0], X2[Y >= 0], s = 80, c = 'b', marker = "o")
    plt.scatter(X1[Y <  0], X2[Y  < 0], s = 80, c = 'r', marker = "^")
    #ax.set_xlim(xl1, xl2)
    #ax.set_ylim(yl1, yl2)
    fig.set_size_inches(6, 6)
    plt.show()


def plot_data_and_line(w1,w2):

    pass

def check_error(w, dataset):
    result = None
    error = 0
    for x, s in dataset:

        x = np.array(x)

        if int(np.sign(w.T.dot(x))) != s:
            print("w.dot(x)=", w.T.dot(x)) #w.T.dot(x) == w.x
            result =  x, s
            print("result=", result)
            error += 1
            return result






    return result

#PLA演算法實作

def pla(dataset):
    #general equation of seperation line: A+BX+CY=0 => W = (1,x,y) => Y = -(A/C) + (-B/C)X
    w = np.zeros(3)


    while True:
        print("After while")
        result = check_error(w, dataset)
        if result is None:
            break
        x,s = result

        w += s * x
    return w




w = pla(dataset)
print(w)






positive = []
negative = []
for v in dataset:
    if v[1] == 1:
        positive.append(v[0])
    else:
        negative.append(v[0])


fig = plt.figure()
ax1 = fig.add_subplot(111)


plt.scatter(X1[Y >= 0], X2[Y >= 0], s = 80, c = 'b', marker = "o")
plt.scatter(X1[Y <  0], X2[Y  < 0], s = 80, c = 'r', marker = "^")

l = np.linspace(-2,2)
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
fig.set_size_inches(6, 6)
plt.show()


from clusterize import *

L = Loader0()
R = np.array((7,7))
for i in range(1,8):
    for j in range(i+1,8):
        X1,X2 = corr_xyz(L,i,j,40,40,40)
        R[i,j]=R[j,i]=pearsonr(X1,X2)
f,ax = plt.subplot((1,1))
ax.plot(R)
plt.show()
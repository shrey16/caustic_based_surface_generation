import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp
from scikits.umfpack import spsolve
import matplotlib.pyplot as plt

N = 512
source = np.loadtxt('source_0.txt')
source = np.reshape(source,(N,N,2))
D = np.loadtxt('D_0.txt')

def flatten(source):
    source_inner = source[1:N-1,1:N-1]
    points_inner = np.reshape(source_inner,(-1,2))
    N_points_inner = len(points_inner)
    points_boundary = np.append(np.append(source[0,:],source[N-1,:],axis=0),np.append(source[1:N-1,0],source[1:N-1,N-1],axis=0),axis=0)
    points = np.append(points_inner,points_boundary,axis=0)
    return points, N_points_inner
points, N_points_inner = flatten(source)
points_del = Delaunay(points)
triangles = points_del.simplices
M = len(triangles)

def stiff():

    def Gradients(x,y):
        Dphi = np.zeros((3,2))
        Dphi[0,0] = y[1] - y[2]
        Dphi[0,1] = x[2] - x[1]
        Dphi[1,0] = y[2] - y[0]
        Dphi[1,1] = x[0] - x[2]
        Dphi[2,0] = y[0] - y[1]
        Dphi[2,1] = x[1] - x[0]
        return Dphi

    index = [[] for idx in range(N_points_inner)]
    val = [[] for idx in range(N_points_inner)]

    for k in range(M):
        if not k%10000: print(k)
        x = points[list(triangles[k]),0]
        y = points[list(triangles[k]),1]
        D = x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1])
        Dphi = Gradients(x,y)
        for ki in range(3):
            i = triangles[k,ki]
            if i < N_points_inner:
                for kj in range(3):
                    j = triangles[k,kj]
                    if j < N_points_inner:
                        try:
                            jdx = index[i].index(j)
                        except:
                            index[i].append(j)
                            val[i].append(0)
                            jdx = index[i].index(j)
                        
                        val[i][jdx] += (Dphi[ki,0]*Dphi[kj,0]+Dphi[ki,1]*Dphi[kj,1])/(D/2)

    row_ind = []
    col_ind = []
    val_ind = []

    for idx in range(len(index)):
        for jdx in range(len(index[idx])):
            row_ind.append(idx)
            col_ind.append(index[idx][jdx])
            val_ind.append(val[idx][jdx])

    S = sp.coo_matrix((val_ind, (row_ind, col_ind)))
    return S

def flattenD(D):
    D_inner = D[1:N-1,1:N-1]
    D_flat_inner = np.reshape(D_inner,-1)
    D_flat_boundary = np.append(np.append(D[0,:],D[N-1,:]),np.append(D[1:N-1,0],D[1:N-1,N-1]))
    D_flat = np.append(D_flat_inner,D_flat_boundary)
    return D_flat

def rhs():
    D_flat = flattenD(D)
    print(np.shape(D_flat))
    def tri_area(x,y):
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    triangle_areas = np.zeros(M)
    for i in range(M):
        x = points[list(triangles[i]),0]
        y = points[list(triangles[i]),1]
        triangle_areas[i] = tri_area(x,y)
    triangles_by_node = [[] for i in range(N*N)]
    for i in range(M):
        for point in triangles[i]:
            triangles_by_node[point].append(i)
    def supp(point):
        area = 0
        for triangle_idx in triangles_by_node[point]:
            area += triangle_areas[triangle_idx]
        return area
    rhs = np.zeros_like(D_flat)
    for i in range(N*N):
        rhs[i] = D_flat[i]*supp(i)/3
    return rhs

def gallerkin():
    S = stiff()
    sp.save_npz('S.npz',S)
    S = S.tocsr()
    g = rhs()
    np.savetxt('g.txt',g)
    phi = spsolve(S,g)
    np.savetxt('phi_galerkin.txt',phi)

gallerkin()
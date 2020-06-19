import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.image as img
import pathlib

curr_location = pathlib.Path(__file__).parent.absolute()
path = curr_location / 'support'
path.mkdir(parents=True, exist_ok=True)
path = str(path)

img_location = str(curr_location)+'/baboon.gif'

def area(points_source,simplices):
    def PolyArea(points, center):
        points = [np.array(point) for point in points]
        def clockwiseangle_and_distance(point):
            vec = point - center
            l = np.linalg.norm(vec)
            return np.arctan2(vec[1],vec[0]), l
        points = sorted(points, key=clockwiseangle_and_distance)
        points = np.array(points)
        x = points[:,0]; y = points[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    areas = np.zeros(512*512)
    ntri_all = [[] for i in range(512*512)]
    for simplex in simplices:
        for a in simplex:
            ntri_all[a].append(simplex)
    for i in range(512*512):
        ntri = ntri_all[i]
        points = set()
        for simplex in ntri:
            simplex = np.delete(simplex,np.argwhere(simplex==i))
            a = points_source[i]
            b = points_source[simplex[0]]
            c = points_source[simplex[1]]
            points.add(tuple((a+b)/2))
            points.add(tuple((a+c)/2))
            points.add(tuple((a+b+c)/3))
        if len(points) == 3:
            points.add(tuple(a))
        points = list(points)
        areas[i] = PolyArea(points, a)
    return areas

def calcDiff(points_source,tri_source,L,C,areas_dest):
    points_source = np.reshape(source,(-1,2))
    tri_source = Delaunay(points_source)
    areas_source = area(points_source,tri_source.simplices)
    J = areas_source/areas_dest
    J = np.reshape(J,(512,512))
    D = L*J - C
    dmin = np.min(D); dmax = np.max(D)
    translation_factor = (dmin + dmax)/2
    D -= translation_factor
    Cavg = np.average(C)
    dmin = np.min(D); dmax = np.max(D)
    scaling_factor = 4*Cavg/(dmax-dmin)
    D *= scaling_factor
    return D

def poisson_solver(D):
    phi = np.zeros_like(D)
    w = 1

    def sor_step(phi,ro,w):
        len_x, len_y = np.shape(phi)
        phi_new = np.copy(phi)
        delta = 1
        for i in range(len_x):
            for j in range(len_y):
                if i==0 or j==0 or i==len_x-1 or j==len_y-1:
                    continue
                phixp = phi_new[i-1,j]
                phixn = phi_new[i+1,j]
                phiyp = phi_new[i,j-1]
                phiyn = phi_new[i,j+1]
                phi_n = (phixp+phixn+phiyp+phiyn)/4 - delta**2*ro[i,j]/4
                phi_new[i,j] = w*phi_n + (1-w)*phi[i,j]
        
        return phi_new

    for i in range(200):
        phi_new = sor_step(phi,-D,w)
        phi = phi_new
    return -phi

def calcDeltaT(points_source,tri_source,delphi):
    points_delphi = np.reshape(delphi, (-1,2))
    points_tri = np.array([[points_source[j] for j in simplex] for simplex in tri_source.simplices])
    delpoints_tri = np.array([[points_delphi[j] for j in simplex] for simplex in tri_source.simplices])
    def isClockwise(a,b,c):
        a = np.append(a,1)
        b = np.append(b,1)
        c = np.append(c,1)
        return np.sign(np.linalg.det(np.vstack((a,b,c))))
    delT = 0
    init_sign_arr = np.array([isClockwise(*tri) for tri in points_tri])
    sign_arr = np.copy(init_sign_arr)
    while (init_sign_arr - sign_arr == 0).all():
        delT += 0.05
        points_tri += delT*delpoints_tri
        sign_arr = np.array([isClockwise(*tri) for tri in points_tri])
    return delT/2

def gradient(phi):
    N, N = np.shape(phi)
    grad_x, grad_y = np.zeros((N,N)), np.zeros((N,N))
    grad_x[:,1:N-1] = phi[:,1:N-1] - phi[:,:N-2]
    grad_y[1:N-1,:] = phi[1:N-1,:] - phi[:N-2,:]
    grad = np.zeros((N,N,2))
    grad[:,:,0] = grad_x; grad[:,:,1] = grad_y
    return grad

def divergence(N_xy):
    N, N, d = np.shape(N_xy)
    div = np.zeros((N,N))
    div[1:,1:] = N_xy[1:,1:,0]-N_xy[:N-1,1:,0] + N_xy[1:,1:,1]-N_xy[1:,:N-1,1]
    div[N-1,:] = np.zeros_like(div[N-1,:])
    div[:,N-1] = np.zeros_like(div[:,N-1])
    return div

image = img.imread(img_location)
imgdata = np.zeros((512,512))
for i in range(512):
    for j in range(512):
        imgdata[i,j] = image[i,j,0]

source = np.zeros((512,512,2))
for i in range(512):
    for j in range(512):
        source[i,j] = np.array([i+j*1e-5,j])

dest = np.copy(source)
inten_dest = C = np.copy(imgdata)/255
inten_source = L = np.average(C)*np.ones((512,512))

print('calculating destination areas')
points_dest = np.reshape(dest,(-1,2))
tri_dest = Delaunay(points_dest)
areas_dest = area(points_dest,tri_dest.simplices)
np.savetxt(path+'/areas_dst.txt',areas_dest)
print('done')

np.savetxt(path+'/delT.txt',np.array([0]))

for i in range(19):
    plt.clf()
    print('start iteration',i)
    points_source = np.reshape(source,(-1,2))
    tri_source = Delaunay(points_source)
    np.savetxt(path+'/source_{i}.txt'.format(i=i),points_source)
    plt.triplot(points_source[:,0],points_source[:,1],tri_source.simplices, linewidth=0.1)
    plt.savefig(path+'/iter_{i}.jpg'.format(i=i))
    print('calculating D')
    D = -calcDiff(points_source,tri_source,L,C,areas_dest)
    np.savetxt(path+'/D_{i}.txt'.format(i=i),D)
    print('done')
    print('calculating phi')
    phi = poisson_solver(D)
    np.savetxt(path+'/phi_{i}.txt'.format(i=i),phi)
    print('done')
    gradphi = gradient(phi)
    print('calculating delta t')
    delT = calcDeltaT(points_source,tri_source,gradphi)
    print('delta t =',delT)
    delTarray = np.loadtxt(path+'/delT.txt')
    delTarray = np.append(delTarray,delT)
    np.savetxt(path+'/delT.txt',delTarray)
    source += delT*gradphi

source = np.loadtxt(path+'/source_13.txt')
source = np.reshape(source,(512,512,2))
h = np.zeros((512,512))
k = np.copy(h)
eta = 1.5
H = 100

print("calculating surface")
for itr in range(16):
    print('running iteration number {itr}'.format(itr=itr))
    for i in range(512):
        for j in range(512):
            k[i,j] = eta*np.sqrt(np.linalg.norm(source[i,j]-dest[i,j])**2 + (H-h[i,j])**2) - (H-h[i,j])

    N_xy = source - dest
    N_xy = np.array([[N_xy[i,j]/k[i,j] for i in range(512)] for j in range(512)])

    div = divergence(N_xy)
    np.savetxt(path+'/div_{itr}.txt'.format(itr=itr),div)
    h = poisson_solver(div)
    np.savetxt(path+'/h_{itr}.txt'.format(itr=itr),h)
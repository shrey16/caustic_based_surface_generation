import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pathlib

curr_location = pathlib.Path(__file__).parent.absolute()
path = curr_location / 'support'
path = str(path)

h = np.loadtxt(path+'/h_15.txt')
h = h[511::-1,:]
source = np.loadtxt(path+'/source_13.txt')
source = np.reshape(source,(512,512,2))
source = source/51.2

point_cloud = np.zeros((512,512,3))
point_cloud[:,:,:2] = source
point_cloud[:,:,2] = h
point_cloud = np.reshape(point_cloud,(-1,3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
downpcd = pcd
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],point_show_normal=False)

distances = downpcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd,o3d.utility.DoubleVector([radius, radius * 2]))

bpa_mesh.remove_degenerate_triangles()
bpa_mesh.remove_duplicated_triangles()
bpa_mesh.remove_duplicated_vertices()

o3d.io.write_triangle_mesh(str(curr_location)+"/bpa_mesh.stl", bpa_mesh)
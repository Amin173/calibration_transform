import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def calculate_transform(xy_points, yz_points, xz_points):
  # Find the centroids of the point sets
  xy_centroid = centroid(xy_points)
  yz_centroid = centroid(yz_points)
  xz_centroid = centroid(xz_points)

  # Calculate the normal vectors of the planes defined by the point sets
  xy_normal = normal_vector(xy_points)
  yz_normal = normal_vector(yz_points)
  xz_normal = normal_vector(xz_points)

  # Calculate the distances of the planes from the origin
  xy_distance = dot_product(xy_normal, xy_centroid)
  yz_distance = dot_product(yz_normal, yz_centroid)
  xz_distance = dot_product(xz_normal, xz_centroid)

  # Define the planes
  xy_plane = (xy_normal, xy_distance)
  yz_plane = (yz_normal, yz_distance)
  xz_plane = (xz_normal, xz_distance)

  # Find the intersection of the planes
  origin = intersect(xy_plane, yz_plane, xz_plane)
  print(f"origin: {origin}")

  # Calculate the transformation matrix
  T = transformation_matrix(origin, xy_normal, yz_normal, xz_normal)

  return T

def centroid(points):
  # Calculate the centroid of the point set
  centroid = np.mean(points, axis=0)

  return centroid

def normal_vector(points):
  # Convert points to a NumPy array
  points = np.array(points)

  # Calculate the centroid of the points
  centroid = np.mean(points, axis=0)

  # Center the points at the origin
  points_centered = points - centroid

  # Calculate the normal vector using the cross product of two non-collinear vectors in the plane
  normal = np.cross(points_centered[0] - points_centered[1], points_centered[0] - points_centered[2])

  # Normalize the normal vector
  normal /= np.linalg.norm(normal)

  # Make sure the normal vector points in the direction of increasing y values
  if normal[1] < 0:
    normal = -normal

  return normal
  
def dot_product(v1, v2):
  # Calculate the dot product of two vectors
  dot = np.dot(v1, v2)

  return dot

# def transformation_matrix(origin, xy_normal, yz_normal, xz_normal):
#   # Calculate the transformation matrix from the origin and the normal vectors of the planes
#   T = np.zeros((4, 4))

#   # Use the Gram-Schmidt process to ensure that the normal vectors are orthonormal
#   xy_normal = xy_normal / np.linalg.norm(xy_normal)
#   yz_normal = yz_normal - np.dot(yz_normal, xy_normal) * xy_normal
#   yz_normal = yz_normal / np.linalg.norm(yz_normal)
#   xz_normal = xz_normal - np.dot(xz_normal, xy_normal) * xy_normal - np.dot(xz_normal, yz_normal) * yz_normal
#   xz_normal = xz_normal / np.linalg.norm(xz_normal)

#   # Assign the normal vectors to the rotation matrix
#   T[:3, :3] = np.vstack((xy_normal, yz_normal, xz_normal)).T
#   T[:3, 3] = origin
#   T[3, 3] = 1

#   return T

def transformation_matrix(origin, xy_normal, yz_normal, xz_normal):
  # Calculate the transformation matrix from the origin and the normal vectors of the planes
  T = np.eye(4)
  T[:3, :3] = np.vstack((yz_normal, xz_normal, xy_normal))
  T[:3, 3] = origin
  return T

def intersect(plane1, plane2, plane3):
  # Calculate the intersection of three planes
  n1 = plane1[0]
  n2 = plane2[0]
  n3 = plane3[0]
  d1 = plane1[1]
  d2 = plane2[1]
  d3 = plane3[1]

  # Calculate the determinants of the matrices
  det_A = np.linalg.det(np.vstack((n1, n2, n3)))
  det_A1 = np.linalg.det(np.vstack((d1*np.ones_like(n1), n2, n3)))
  det_A2 = np.linalg.det(np.vstack((n1, d2*np.ones_like(n2), n3)))
  det_A3 = np.linalg.det(np.vstack((n1, n2, d3*np.ones_like(n3))))

  # Calculate the intersection point
  x = det_A1/det_A
  y = det_A2/det_A
  z = det_A3/det_A
  intersection = np.array([x, y, z])

  return intersection

def test_normal_vector():
  # Test 1: Check normal vector of a plane with points on the xy plane
  points = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
  normal = normal_vector(points)
  assert np.allclose(normal, [0, 0, 1], rtol=1e-7, atol=1e-7), f'Error: {normal}'

  # Test 2: Check normal vector of a plane with points on the yz plane
  points = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
  normal = normal_vector(points)
  assert np.allclose(normal, [1, 0, 0], rtol=1e-7, atol=1e-7), f'Error: {normal}'

  # Test 3: Check normal vector of a plane with points on the xz plane
  points = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
  normal = normal_vector(points)
  assert np.allclose(normal, [0, 1, 0], rtol=1e-7, atol=1e-7), f'Error: {normal}'

  # Test 4: Check normal vector of a plane with points on a diagonal
  points = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
  normal = normal_vector(points)
  assert np.allclose(normal, np.array([1, 1, 1])/np.linalg.norm([1, 1, 1]), rtol=1e-7, atol=1e-7), f'Error: {normal}'

  print('normal_vector tests pass')

def test_calculate_transform():

  def get_transform(euler_angles, translation):
    T = np.eye(4)
    r= R.from_euler('zyx', euler_angles, degrees=True)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = translation
    return T

  def transform_points(points, T):
    return (np.dot(T[:3, :3], points) + T[:3, 3])

  def plot_plane(ax, point, normal, color='yellow'):
    # Generate a set of points that lie on the plane
    X, Y = np.meshgrid(range(-5, 6), range(-5, 6))
    if normal[2] != 0:
        # Compute the constants a, b, c, and d in the equation of the plane
        a, b, c = normal
        d = -np.dot(normal, point)
        Z = (-d - a * X - b * Y) * 1. / c
    else:
        Z = np.full_like(X, point[2])

    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.5, color=color)

  def visualize_points(xy_points, yz_points, xz_points):
    # Convert the points to NumPy arrays
    xy_points = np.array(xy_points)
    yz_points = np.array(yz_points)
    xz_points = np.array(xz_points)

    # Set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(xy_points[:, 0], xy_points[:, 1], xy_points[:, 2], c='r', marker='o', alpha=0.2)
    ax.scatter(yz_points[:, 0], yz_points[:, 1], yz_points[:, 2], c='g', marker='x', alpha=0.2)
    ax.scatter(xz_points[:, 0], xz_points[:, 1], xz_points[:, 2], c='b', marker='*', alpha=0.2)

    # Find the centroids of the point sets
    xy_centroid = centroid(xy_points)
    yz_centroid = centroid(yz_points)
    xz_centroid = centroid(xz_points)

    # Calculate the normal vectors of the planes defined by the point sets
    xy_normal = normal_vector(xy_points)
    yz_normal = normal_vector(yz_points)
    xz_normal = normal_vector(xz_points)

    # Plot the planes
    plot_plane(ax, xy_centroid, xy_normal, color='red')
    plot_plane(ax, yz_centroid, yz_normal, color='green')
    plot_plane(ax, xz_centroid, xz_normal, color='blue')

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-5, 5])

    # Show the plot
    plt.show()

  xy_points1 = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
  yz_points1 = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
  xz_points1 = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]

  # Test 1
  visualize_points(xy_points1, yz_points1, xz_points1)
  T = calculate_transform(xy_points1, yz_points1, xz_points1)
  T1 = np.eye(4)
  assert np.allclose(T, T1, rtol=1e-7, atol=1e-7), f'Error: {T}'

  # # Test 2
  # T2 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0, 2],
  #       [np.sin(np.pi/4), np.cos(np.pi/4), 0, 2],
  #       [0, 0, 1, 3],
  #       [0, 0, 0, 1]])
  # xy_points2 = transform_points(xy_points, T2)
  # yz_points2 = transform_points(yz_points, T2)
  # xz_points2 = transform_points(xz_points, T2)
  # visualize_points(xy_points2, yz_points2, xz_points2)
  # T = calculate_transform(xy_points2, yz_points2, xz_points2)
  # assert np.allclose(T, T2, rtol=1e-7, atol=1e-7), f'Error: {T}'
  
  # Test 3
  T3 = get_transform([0, 40, 50], [1, 2, 3])
  print(T3)
  xy_points3 = transform_points(xy_points1, T3)
  yz_points3 = transform_points(yz_points1, T3)
  xz_points3 = transform_points(xz_points1, T3)
  visualize_points(xy_points3, yz_points3, xz_points3)
  T = calculate_transform(xy_points3, yz_points3, xz_points3)
  assert np.allclose(T, T3, rtol=1e-7, atol=1e-7), f'Error: {T}'

  # Test 4
  T4 = np.array([[np.cos(np.pi/4), 0, np.sin(np.pi/4), 1],
        [0, 1, 0, 2],
        [-np.sin(np.pi/4), 0, np.cos(np.pi/4), 3],
        [0, 0, 0, 1]])
  xy_points4 = transform_points(xy_points1, T4)
  yz_points4 = transform_points(yz_points1, T4)
  xz_points4 = transform_points(xz_points1, T4)
  visualize_points(xy_points4, yz_points4, xz_points4)
  T = calculate_transform(xy_points4, yz_points4, xz_points4)
  assert np.allclose(T, T4, rtol=1e-7, atol=1e-7), f'Error: {T}'

  print('calculate_transform tests pass')


if __name__ == '__main__':
  test_normal_vector()
  test_calculate_transform()
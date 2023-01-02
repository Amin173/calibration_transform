import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager
from transform import *

def test_normal_vector():
    # Test 1: Check normal vector of a plane with points on the xy plane
    points = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(normal, [0, 0, 1], rtol=1e-7,
                       atol=1e-7), f'Error: {normal}'

    # Test 2: Check normal vector of a plane with points on the yz plane
    points = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(normal, [1, 0, 0], rtol=1e-7,
                       atol=1e-7), f'Error: {normal}'

    # Test 3: Check normal vector of a plane with points on the xz plane
    points = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(normal, [0, 1, 0], rtol=1e-7,
                       atol=1e-7), f'Error: {normal}'

    # Test 4: Check normal vector of a plane with points on a diagonal
    points = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(normal, np.array(
        [1, 1, 1])/np.linalg.norm([1, 1, 1]), rtol=1e-7, atol=1e-7), f'Error: {normal}'

    print('normal_vector tests pass')


def test_calculate_transform():

    # def transform_points(points, T):
    #     # Convert the points to a homogeneous representation
    #     points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

    #     # Transform the points using the inverse of T
    #     points_homogeneous_transformed = np.linalg.inv(
    #         T) @ points_homogeneous.T

    #     # Convert the points back to a non-homogeneous representation
    #     points_transformed = points_homogeneous_transformed[:3,
    #                                                         :] / points_homogeneous_transformed[3, :]

    #     return points_transformed.T

    def transform_points(points, T):
        # Convert the points to a homogeneous representation
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

        # Transform the points using the inverse of T
        points_homogeneous_transformed = T @ points_homogeneous.T

        # Convert the points back to a non-homogeneous representation
        points_transformed = points_homogeneous_transformed[:3,
                                                            :] / points_homogeneous_transformed[3, :]

        return points_transformed.T


    def plot_plane(ax, plane, color='yellow'):
        # Generate a set of points that lie on the plane
        a, b, c, d = plane
        if c != 0:
            # Compute the constants a, b, c, and d in the equation of the plane
            X, Y = np.meshgrid(range(-5, 5), range(-5, 5))
            Z = (-d - a * X - b * Y) * 1. / c
        elif abs(b)>=abs(a):
            X, Z = np.meshgrid(range(-5, 5), range(-5, 5))
            Y = (-a * X - c * Z - d)/b
        elif a != 0:
            Y, Z = np.meshgrid(range(-5, 5), range(-5, 5))
            X = (-b * Y - c * Z - d)/a
        else:
            raise ValueError(
                "Error: plane coefficents a, b, c can not all be zero")

        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=0.5, color=color)

    def visualize_points(xy_points, yz_points, xz_points):
        # Convert the points to NumPy arrays
        xy_points = np.array(xy_points)
        yz_points = np.array(yz_points)
        xz_points = np.array(xz_points)

        # Calculate the normal vectors of the planes defined by the point sets
        xy_plane = plane_from_points(xy_points)
        yz_plane = plane_from_points(yz_points)
        xz_plane = plane_from_points(xz_points)

        # Set up the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points
        ax.scatter(xy_points[:, 0], xy_points[:, 1],
                   xy_points[:, 2], c='r', marker='o', alpha=0.2)
        ax.scatter(yz_points[:, 0], yz_points[:, 1],
                   yz_points[:, 2], c='g', marker='x', alpha=0.2)
        ax.scatter(xz_points[:, 0], xz_points[:, 1],
                   xz_points[:, 2], c='b', marker='*', alpha=0.2)

        # Plot the planes
        # plot_plane(ax, xy_plane, color='red')
        # plot_plane(ax, yz_plane, color='green')
        # plot_plane(ax, xz_plane, color='blue')
        plt.axis('equal')

        # Set the axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

        # Show the plot
        plt.show()

    xy_points1 = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    yz_points1 = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    xz_points1 = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]

    # Test 1
    #   visualize_points(xy_points1, yz_points1, xz_points1)
    #   T = calculate_transform(xy_points1, yz_points1, xz_points1)
    #   T1 = np.eye(4)
    #   assert np.allclose(T, T1, rtol=1e-7, atol=1e-7), f'Error: {T}'

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
    #   print(f"\n\n Test 3")
    #   T3 = transformation_matrix(origin=[1, 2, 3], roll=0, pitch=40, yaw=50)
    #   print(T3)
    #   xy_points3 = transform_points(xy_points1, T3)
    #   yz_points3 = transform_points(yz_points1, T3)
    #   xz_points3 = transform_points(xz_points1, T3)
    #   visualize_points(xy_points3, yz_points3, xz_points3)
    #   T = calculate_transform(xy_points3, yz_points3, xz_points3)
    #   assert np.allclose(T, T3, rtol=1e-7, atol=1e-7), f'Error: {T}'

    # Test 4

    T4 = transformation_matrix(origin=[1, 2, 3], roll=10, pitch=10, yaw=10)
    T_initial = transformation_matrix(origin=[0,0,0], roll=0, pitch=0, yaw=0)

    xy_points4 = transform_points(xy_points1, T4)
    yz_points4 = transform_points(yz_points1, T4)
    xz_points4 = transform_points(xz_points1, T4)
    visualize_points(xy_points4, yz_points4, xz_points4)
    T = calculate_transform(xy_points4, yz_points4, xz_points4, T_initial)

    assert np.allclose(
        T, T4, rtol=1e-7, atol=1e-7), f'\n\n T: {T}\n\n T_true: {T4}\n'

    print('calculate_transform tests pass')


if __name__ == '__main__':
    test_normal_vector()
    test_calculate_transform()

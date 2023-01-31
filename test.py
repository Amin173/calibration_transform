import matplotlib.pyplot as plt
from transform import *
from scipy.spatial.transform import Rotation as R
import csv
import json 

def get_points_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    xy_points = []
    yz_points = []
    xz_points = []
    for point in data:
        if point['description'] == 'xy':
            xy_points.append([point["coordinates"]['x']["mu"], point["coordinates"]['y']["mu"], point["coordinates"]['z']["mu"]])
        elif point['description'] == 'yz':
            yz_points.append([point["coordinates"]['x']["mu"], point["coordinates"]['y']["mu"], point["coordinates"]['z']["mu"]])
        elif point['description'] == 'xz':
            xz_points.append([point["coordinates"]['x']["mu"], point["coordinates"]['y']["mu"], point["coordinates"]['z']["mu"]])
    return xy_points, yz_points, xz_points

def test_normal_vector():
    # Test 1: Check normal vector of a plane with points on the xy plane
    points = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(np.cross(normal, [0, 0, 1]), 0, rtol=1e-7,
                       atol=1e-7), f'Error: {normal}'

    # Test 2: Check normal vector of a plane with points on the yz plane
    points = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(np.cross(normal, [1, 0, 0]), 0, rtol=1e-7,
                       atol=1e-7), f'Error: {normal}'

    # Test 3: Check normal vector of a plane with points on the xz plane
    points = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(np.cross(normal, [0, 1, 0]), 0, rtol=1e-7,
                       atol=1e-7), f'Error: {normal}'

    # Test 4: Check normal vector of a plane with points on a diagonal
    points = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    plane_coefficients = plane_from_points(points)
    normal = plane_coefficients[:3]
    assert np.allclose(np.cross(normal, np.array(
        [1, 1, 1])/np.linalg.norm([1, 1, 1])), 0, rtol=1e-7, atol=1e-7), f'Error: {normal}'

    print('plane normal tests pass')


def transform_points(points, T):
    # Convert the points to a homogeneous representation
    points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

    # Transform the points using the inverse of T
    points_homogeneous_transformed = T @ points_homogeneous.T

    # Convert the points back to a non-homogeneous representation
    points_transformed = points_homogeneous_transformed[:3,
                                                        :] / points_homogeneous_transformed[3, :]

    return points_transformed.T

def plot_plane(ax, points, color='yellow'):

    # Calculate the plane coefficients for the set of points
    a, b, c, d = plane_from_points(points)

    # Compute the constants a, b, c, and d in the equation of the plane
    if c != 0:
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

def visualize_points(xy_points, yz_points, xz_points, show_planes=False):
    # Convert the points to NumPy arrays
    xy_points = np.array(xy_points)
    yz_points = np.array(yz_points)
    xz_points = np.array(xz_points)

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

    if show_planes:
        # Plot the planes
        plot_plane(ax, xy_points, color='red')
        plot_plane(ax, yz_points, color='green')
        plot_plane(ax, xz_points, color='blue')

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the axis limits
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    # set the axes scales equal
    plt.axis('equal')

    # Show the plot
    plt.show()

def test_calculate_transform():


    # Test 1
    T1 = np.eye(4)
    xy_points1 = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    yz_points1 = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    xz_points1 = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    # visualize_points(xy_points1, yz_points1, xz_points1)
    T = calculate_transform(xy_points1, yz_points1, xz_points1)
    assert np.allclose(T, T1, rtol=1e-7, atol=1e-7), f'Error: {T}'

    # # Test 2
    T4 = transformation_matrix(origin=[1, 2, 3], euler_angles = (10, 10, 200))
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 0, 180))
    xy_points4 = transform_points(xy_points1, T4)
    yz_points4 = transform_points(yz_points1, T4)
    xz_points4 = transform_points(xz_points1, T4)
    # visualize_points(xy_points4, yz_points4, xz_points4)
    T = calculate_transform(xy_points4, yz_points4, xz_points4, T_initial)
    assert np.allclose(
        T, T4, rtol=1e-7, atol=1e-7), f'\n\n T: {T}\n\n T_true: {T4}\n'

    # Test 3
    T4 = transformation_matrix(origin=[1, 2, 3], euler_angles = (10, 100, 80))
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 90, 90))
    xy_points4 = transform_points(xy_points1, T4)
    yz_points4 = transform_points(yz_points1, T4)
    xz_points4 = transform_points(xz_points1, T4)
    # visualize_points(xy_points4, yz_points4, xz_points4)
    T = calculate_transform(xy_points4, yz_points4, xz_points4, T_initial)
    assert np.allclose(
        T, T4, rtol=1e-7, atol=1e-7), f'\n\n T: {T}\n\n T_true: {T4}\n'

    # Test 4
    T4 = transformation_matrix(origin=[1, 2, 3], euler_angles = (10, 10, -10))
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 0, 0))
    xy_points4 = transform_points(xy_points1, T4)
    yz_points4 = transform_points(yz_points1, T4)
    xz_points4 = transform_points(xz_points1, T4)
    # visualize_points(xy_points4, yz_points4, xz_points4)
    T = calculate_transform(xy_points4, yz_points4, xz_points4, T_initial)
    assert np.allclose(
        T, T4, rtol=1e-7, atol=1e-7), f'\n\n T: {T}\n\n T_true: {T4}\n'


    # Test 4
    T4 = transformation_matrix(origin=[0, 0, 0], euler_angles = (10, 10, -10))
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 0, 0))
    xy_points4 = transform_points(xy_points1, T4)
    yz_points4 = transform_points(yz_points1, T4)
    xz_points4 = transform_points(xz_points1, T4)
    # visualize_points(xy_points4, yz_points4, xz_points4)
    T = calculate_transform(xy_points4, yz_points4, xz_points4, T_initial)
    assert np.allclose(
        T, T4, rtol=1e-7, atol=1e-7), f'\n\n T: {T}\n\n T_true: {T4}\n'

    print('full transform tests pass')

def test_real_data():
    print("\n\n")
    print("--------------------------------")
    print("Real data test results: ")
    print("--------------------------------")

    # Test 1
    T1 = transformation_matrix(origin=[-0.815,	1.795,	-2.35], euler_angles = (0, 0, 0.0039*180/np.pi))
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 0, 0))
    xy_points1 = np.array([[47.864, 32.161, -2.96], [47.864, 271.839, -2.12], [185.649, 271.839, -1.74], [185.649, 32.161, -2.58]])
    xz_points1 = np.array([[34.83,	1.9,	-12.70], [183.315, 2.37, -12.70], [109.0725,	2.135,	0]])
    yz_points1 = np.array([[-1.01,	32.909,	-12.70], [-2.34,	270.754,	-12.70], [-1.675,	151.8315,	0]])
    T = calculate_transform(xy_points1, yz_points1, xz_points1, T_initial)
    euler_angles = R.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=True)

    print(f"New transform: {T}\n")
    print(f"Old transform: {T1}\n\n")

    print(f"New origin and euler angles: origin[mm] = [{T[:3, 3]}] euler angels[deg]: {euler_angles}\n")
    print(f"Old origin and euler angles: origin[mm] = [{[-0.815,	1.795,	-2.35]}] euler angels[deg]: {(0, 0, 0.0039*180/np.pi)}\n\n")

def test_real_data_2(file_path):
    print("\n\n")
    print("--------------------------------")
    print("Real data test results: ")
    print("--------------------------------")

    # Test 1
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 0, -90))
    xy_points1, yz_points1, xz_points1 = get_points_from_json(file_path)
    T = calculate_transform(xy_points1, yz_points1, xz_points1, T_initial)
    
    euler_angles = R.from_matrix(T[:3, :3]).as_euler('XYZ', degrees=True)

    print(f"New transform: {T}\n")
    print(f"Old transform: {T_initial}\n\n")

    print(f"New origin and euler angles: origin[mm] = [{T[:3, 3]}] euler angels[deg]: {euler_angles}\n")
    print(f"Old origin and euler angles: origin[mm] = [{[-0.815,	1.795,	-2.35]}] euler angels[deg]: {(0, 0, -90)}\n\n")

def test_fake_data(file_path):
    print("\n\n")
    print("--------------------------------")
    print("Fake data test results: ")
    print("--------------------------------")

    # Initialize lists to store the points for each plane
    xy_points = []
    yz_points = []
    xz_points = []
    T_initial = transformation_matrix(origin=[0,0,0], euler_angles = (0, 0, 0))
    # Read the CSV file and extract the points for each plane
    with open(file_path, 'r') as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            # Skip the first row
            if i == 0:
                continue

            # Extract the x, y, z, and plane label from the row
            x, y, z, plane_label = row
            x, y, z = float(x), float(y), float(z)

            # Append the point to the appropriate list
            if plane_label.lower() == 'xy':
                xy_points.append([x, y, z])
            elif plane_label.lower() == 'yz':
                yz_points.append([x, y, z])
            elif plane_label.lower() == 'xz':
                xz_points.append([x, y, z])

    # Calculate the transformation matrix
    T = calculate_transform(xy_points, yz_points, xz_points, T_initial)
    euler_angles = R.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=True)

    print(f"Transform: {T}\n")
    print(f"origin[mm] = [{T[:3, 3]}] ")
    print(f"euler angels[deg]: {euler_angles}\n")

if __name__ == '__main__':
    # test_normal_vector()
    # test_calculate_transform()
    # test_real_data()

    # test_fake_data("./test_data/TestPoints_LowNoise.csv")
    # test_fake_data("./test_data/TestPoints_HighNoise.csv")

    test_real_data_2("./test_data/measurements-1.json")
    test_real_data_2("./test_data/measurements-2.json")
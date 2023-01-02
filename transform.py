import numpy as np


def calculate_transform(xy_points, yz_points, xz_points, T_intial):
    # Define the planes
    xy_plane = plane_from_points(xy_points)
    yz_plane = plane_from_points(yz_points)
    xz_plane = plane_from_points(xz_points)

    # Calculate the coordidate system 3d pose parameters
    origin, roll, pitch, yaw = coordinate_system_euler(
        xy_plane, yz_plane, xz_plane, T_intial)

    print(
        f"origin: {origin} *** roll: {roll}  ***  pitch: {pitch}  ***  yaw: {yaw}")

    # Calculate the transformation matrix
    T = transformation_matrix(origin, roll, pitch, yaw)

    return T


def intersection(plane1, plane2, plane3):
    # Extract the coefficients of the three planes
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    a3, b3, c3, d3 = plane3

    # Solve the system of linear equations formed by the three planes
    # to find the point of intersection (x, y, z)
    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    b = np.array([d1, d2, d3])

    # Compute the least square solution of Ax = b
    (x, y, z) = np.linalg.lstsq(A, b)[0]

    # Return the point of intersection as a tuple
    return (x, y, z)

def orientation_to_euler(x_axis, y_axis, z_axis):
    # Normalize the axes
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Compute the roll angle
    roll = np.arctan2(y_axis[2], z_axis[2])

    # Compute the pitch angle
    pitch = np.arcsin(-x_axis[2])

    # Compute the yaw angle
    yaw = np.arctan2(x_axis[1], x_axis[0])

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def calculate_axes(plane1, plane2, plane3, T_intial):
    # Compute the normal vector of( the xy plane
    n1 = np.abs(plane1[:3]) * np.sign((T_intial @ [0, 0, 1, 1])[:3]@[0, 0, 1])

    # Compute the normal vector of the yz plane
    n2 = np.abs(plane2[:3]) *  np.sign((T_intial @ [1, 0, 0, 1])[:3]@[1, 0, 0])

    # Compute the normal vector of the xz plane
    n3 = np.abs(plane3[:3]) *  np.sign((T_intial @ [0, 1, 0, 1])[:3]@[0, 1, 0])

    # Compute the x axis as the cross product of the yz and xz planes
    x_axis = np.cross(n3, n1)

    # Compute the y axis as the cross product of the xy and xz planes
    y_axis = np.cross(n1, n2)

    # Compute the z axis as the cross product of the xy and yz planes
    z_axis = np.cross(n2, n3)

    # Return the axes
    return x_axis, y_axis, z_axis


def coordinate_system_euler(plane1, plane2, plane3, T_intial):
    # Find the intersection point of the three planes
    origin = intersection(plane1, plane2, plane3)
    origin = np.array(origin)

    x_axis, y_axis, z_axis = calculate_axes(plane1, plane2, plane3, T_intial)

    # Compute the roll, pitch, and yaw angles
    roll, pitch, yaw = orientation_to_euler(x_axis, y_axis, z_axis)

    return origin, roll, pitch, yaw


def plane_from_points(points):
    if len(points) < 3:
        raise ValueError("Error: the number of points must be three or more.")

    if len(points) == 3:

        # Extract the coordinates of the three points
        x1, y1, z1 = points[0]
        x2, y2, z2 = points[1]
        x3, y3, z3 = points[2]

        # Check if the points are collinear
        area = 0.5 * \
            np.linalg.norm(
                np.cross((x2-x1, y2-y1, z2-z1), (x3-x1, y3-y1, z3-z1)))
        if area == 0:
            raise ValueError("Error: points are collinear")

        # Compute the coefficients of the plane's equation
        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
    else:
        # Convert the points to a matrix
        points = np.array(points)

        # Create a matrix of the form [x, y, z, 1]
        A = np.hstack([points, np.ones((len(points), 1))])

        # Perform a singular value decomposition of the matrix
        U, S, Vt = np.linalg.svd(A)

        # The last column of Vt is the solution to the least-squares problem
        a, b, c, d = Vt[-1, :]

    # Normalize the coefficients
    norm = np.sqrt(a**2 + b**2 + c**2)
    a /= norm
    b /= norm
    c /= norm
    d /= norm

    # Make sure the normal vector points in the direction of increasing y values
    if b < 0:
        a, b, c, d = -np.array([a, b, c, d])

    return (a, b, c, d)


def transformation_matrix(origin, roll, pitch, yaw):
    # Calculate the transformation matrix from the origin and the normal vectors of the planes
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation(roll, pitch, yaw)
    T[:3, 3] = origin
    return T


def euler_to_rotation(roll, pitch, yaw):
    # Convert the angles to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Compute the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = R_x @ R_y @ R_z

    return R

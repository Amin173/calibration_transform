import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_transform(xy_points, yz_points, xz_points, T_intial=np.eye(4)):
    # Define the planes
    xy_plane = plane_from_points(xy_points)
    yz_plane = plane_from_points(yz_points)
    xz_plane = plane_from_points(xz_points)

    # Find the intersection point of the three planes
    origin = intersection(xy_plane, yz_plane, xz_plane)
    origin = np.array(origin)

    # Calculate the rotation matrix
    rot_mat = coordinate_system_rot_matrix(xy_plane, yz_plane, xz_plane, T_intial)

    # Calculate the full transformation matrix
    T = transformation_matrix(origin, rot_mat=rot_mat)

    # Return the transformation matrix
    return T


def plane_from_points(points):
    if len(points) < 3:
        raise ValueError("Error: the number of points must be three or more.")

    # Convert the points to a matrix
    points = np.array(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Subtract the centroid from each point
    points_cent = points - centroid

    # Calculate the SVD of the centered points
    U, S, Vt = np.linalg.svd(points_cent)

    # The last column of Vt is the solution to the least-squares problem
    a, b, c = Vt[-1, :]

    # Calculate the constant term D
    d = -(a*centroid[0] + b*centroid[1] + c*centroid[2])

    # Normalize the coefficients
    norm = np.sqrt(a**2 + b**2 + c**2)
    a /= norm
    b /= norm
    c /= norm
    d /= norm

    return (a, b, c, d)


def intersection(plane1, plane2, plane3):
    # Extract the coefficients of the three planes
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    a3, b3, c3, d3 = plane3

    # Solve the system of linear equations formed by the three planes
    # to find the point of intersection (x, y, z)
    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    b = -np.array([d1, d2, d3])

    # Compute the least square solution of Ax = b
    (x, y, z) = np.linalg.lstsq(A, b, rcond=None)[0]

    # Return the point of intersection as a tuple
    return (x, y, z)


def coordinate_system_euler(plane1, plane2, plane3, T_intial):

    x_axis, y_axis, z_axis = calculate_axes(plane1, plane2, plane3, T_intial)

    # Compute the roll, pitch, and yaw angles
    roll, pitch, yaw = orientation_to_euler(x_axis, y_axis, z_axis)

    # Return the roll, pitch, and yaw angles
    return roll, pitch, yaw


def coordinate_system_rot_matrix(plane1, plane2, plane3, T_intial):

    x_axis, y_axis, z_axis = calculate_axes(plane1, plane2, plane3, T_intial)

    # Compute the roll, pitch, and yaw angles
    rot_mat = np.vstack((x_axis, y_axis, z_axis)).T

    # Return the roll, pitch, and yaw angles
    return rot_mat


def calculate_axes(plane1, plane2, plane3, T_intial):
    # Compute the plane normals
    n1 = np.array(plane1[:3])
    n2 = np.array(plane2[:3]) 
    n3 = np.array(plane3[:3])

    # fix the plane normal directions using the T_init guess   
    n1 *= np.sign((T_intial @ [0, 0, 1, 1])[:3] @ n1)
    n2 *= np.sign((T_intial @ [1, 0, 0, 1])[:3] @ n2)
    n3 *= np.sign((T_intial @ [0, 1, 0, 1])[:3] @ n3)

    # Compute the x axis as the cross product of the yz and xz planes
    x_axis = np.cross(n3, n1)
    x_axis = n2

    # Compute the y axis as the cross product of the xy and xz planes
    y_axis = np.cross(n1, n2)
    y_axis = n3

    # Compute the z axis as the cross product of the xy and yz planes
    z_axis = np.cross(n2, n3)
    z_axis = n1

    # Normalize the axes
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Return the axes
    return x_axis, y_axis, z_axis


def orientation_to_euler(x_axis, y_axis, z_axis):

    # Compute the roll angle
    roll = np.arctan2(y_axis[2], z_axis[2])

    # Compute the pitch angle
    pitch = np.arcsin(-x_axis[2])

    # Compute the yaw angle
    yaw = np.arctan2(x_axis[1], x_axis[0])

    r = R.from_matrix(np.vstack((x_axis, y_axis, z_axis)))
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    # Return the roll, pitch, and yaw angles
    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def transformation_matrix(origin, rot_mat=None, euler_angles=None):
    # Calculate the transformation matrix from the origin and the normal vectors of the planes
    T = np.eye(4)
    if not isinstance(rot_mat, type(None)):
        T[:3, :3] = rot_mat
    elif not isinstance(euler_angles, type(None)):
        roll, pitch, yaw = euler_angles
        T[:3, :3] = euler_to_rotation(roll, pitch, yaw)
    else:
        raise ValueError("Error: either euler_angles or rot_mat must be specified")
    T[:3, 3] = origin

    # Return the transformation matrix
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

    # Return the rotation matrix
    return R

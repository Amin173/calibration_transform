#/usr/bin/pyhon3

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor

def calculate_transform(xy_points, yz_points, xz_points, R_initial=np.eye(3)):
    """
    Calculate transformation matrix from three planes to parent coordinate system.
    
    Parameters:
    ________________________________________________________________
    xy_points: list
        3D points on xy plane.
    yz_points: list
        3D points on yz plane.
    xz_points: list
        3D points on xz plane.
    T_initial: np.ndarray, optional
        Initial transformation matrix from parent to child coordinate system. Default is identity matrix.
    
    Returns:
    ________________________________________________________________
    np.ndarray:
        Transformation matrix from parent to child coordinate system.
    """
    # Define planes
    xy_plane = plane_from_points(xy_points)
    yz_plane = plane_from_points(yz_points)
    xz_plane = plane_from_points(xz_points)

    # Find intersection of planes
    origin = intersection(xy_plane, yz_plane, xz_plane)
    origin = np.array(origin)

    # Calculate rotation matrix
    rot_mat = coordinate_system_rot_matrix(xy_plane, yz_plane, xz_plane, R_initial)

    # Calculate full transformation matrix
    T = transformation_matrix(origin, rot_mat=rot_mat)

    return T


def plane_from_points(points, inlier_threshold=0.5):
    """
    Calculate plane coefficients from 3D points on plane.
    
    Parameters:
    ________________________________________________________________
    points: list
        3D points on plane.
    inlier_threshold: float, optional
        Inlier threshold. Default is 0.5[mm].
    
    Returns:
    ________________________________________________________________
    tuple: 
        Coefficients (a, b, c, d) of plane.
    """
    if len(points) < 3:
        raise ValueError("Error: the number of points must be three or more.")

    # Convert the points to a matrix
    points = np.array(points)

    # Create a RANSAC regressor
    ransac = RANSACRegressor(estimator=LinearRegression(), 
                            max_trials=300, 
                            min_samples=3, 
                            loss='absolute_error', 
                            residual_threshold=inlier_threshold, 
                            random_state=0)

    # Fit the model to the data
    ransac.fit(points[:, :2], points[:, 2])

    # Get the inlier mask
    inlier_mask = ransac.inlier_mask_

    # Use the inliers to estimate the plane equation
    inlier_points = points[inlier_mask]

    # Calculate the centroid of the points
    centroid = np.mean(inlier_points, axis=0)

    # Subtract the centroid from each point
    points_cent = inlier_points - centroid

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
    """
    Calculate intersection of three planes.
    
    Parameters:
    ________________________________________________________________
    plane1: tuple
        Coefficients (a, b, c, d) of first plane".
    plane2: tuple 
        Coefficients (a, b, c, d) of second plane".
    plane3: tuple
        Coefficients (a, b, c, d) of third plane".
    
    Returns:
    ________________________________________________________________
    tuple: 
        Coordinates (x, y, z) of intersection point.
    """
    # Extract plane coefficients
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    a3, b3, c3, d3 = plane3

    # Solve system of linear equations to find intersection point (x, y, z)
    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    b = -np.array([d1, d2, d3])
    (x, y, z) = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return (x, y, z)


def coordinate_system_rot_matrix(plane1, plane2, plane3, R_initial):
    """
    Calculate rotation matrix for plane coordinate system.
    
    Parameters:
    ________________________________________________________________
    plane1: tuple
        Coefficients (a, b, c, d) of first plane".
    plane2: tuple
        Coefficients (a, b, c, d) of second plane".
    plane3: tuple
        Coefficients (a, b, c, d) of third plane".
    T_initial: np.ndarray
        Initial transformation matrix from world to plane coordinate system.
    
    Returns:
    ________________________________________________________________
    np.ndarray: 
        3x3 rotation matrix for plane coordinate system.
    """
    x_axis, y_axis, z_axis = calculate_axes(plane1, plane2, plane3, R_initial)

    # Compute the roll, pitch, and yaw angles
    rot_mat = np.vstack((x_axis, y_axis, z_axis)).T

    # Return the roll, pitch, and yaw angles
    return rot_mat


def calculate_axes(plane1, plane2, plane3, R_initial):
    """
    Calculate coordinate system axes from plane normals.
    
    Parameters:
    ________________________________________________________________
    plane1: tuple
        Coefficients (a, b, c, d) of first plane".
    plane2: tuple
        Coefficients (a, b, c, d) of second plane".
    plane3: tuple
        Coefficients (a, b, c, d) of third plane".
    T_initial: np.ndarray
        Initial transformation matrix from world to plane coordinate system.
    
    Returns:
    ________________________________________________________________
    tuple:
        Coordinate system axes as row vectors.
    """
    # Compute plane normals
    n1 = np.array(plane1[:3])
    n2 = np.array(plane2[:3]) 
    n3 = np.array(plane3[:3])

    # Fix plane normal directions using R_initial guess
    n1 *= np.sign((R_initial @ [0, 0, 1]) @ n1)
    n2 *= np.sign((R_initial @ [1, 0, 0]) @ n2)
    n3 *= np.sign((R_initial @ [0, 1, 0]) @ n3)

    # Normalize and orthogonalize plane normals (fix numerical errors)
    n1, n2, n3 = gram_schmidt(n1, n2, n3)

    # Compute axes
    x_axis = n2
    y_axis = n3
    z_axis = n1

    return (x_axis, y_axis, z_axis)


def gram_schmidt(n1, n2, n3):
    """
    Normalize and orthogonalize plane normals using the Gram-Schmidt process.
    
    Parameters:
    ________________________________________________________________
    n1: np.ndarray
        Normal vector for first plane.
    n2: np.ndarray
        Normal vector for second plane.
    n3: np.ndarray
        Normal vector for third plane.
    
    Returns:
    ________________________________________________________________
    tuple
        Orthogonalized plane normals.
    """
    # Initialize list of input vectors
    vectors = [n1, n2, n3]
    
    # Orthogonalize vectors
    for i, v in enumerate(vectors):
        for j in range(i):
            v -= v.dot(vectors[j]) * vectors[j]
        v /= v.dot(v)**0.5

    return vectors


def transformation_matrix(origin, rot_mat=None, euler_angles=None):
    """
    Calculate 4x4 transformation matrix from origin and orientation.
    
    Parameters:
    ________________________________________________________________
    origin: np.ndarray
        Translation vector for transformation.
    rot_mat: np.ndarray
        Rotation matrix for transformation.
    euler_angles: tuple, not used if rot_mat is not None
        Tuple of Euler angles for transformation.
    
    Returns:
    ________________________________________________________________
    np.ndarray
        4x4 transformation matrix.
    """
    T = np.eye(4)
    if rot_mat is not None:
        T[:3, :3] = rot_mat
    elif euler_angles is not None:
        roll, pitch, yaw = euler_angles
        T[:3,:3] = euler_to_rotation(roll, pitch, yaw)
    else:
        raise ValueError("Error: either euler_angles or rot_mat must be specified")
    T[:3, 3] = origin

    return T


def euler_to_rotation(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix.
    
    Parameters:
    ________________________________________________________________
    roll: float
        Rotation about x-axis in degrees.
    pitch: float
        Rotation about y-axis in degrees.
    yaw: float
        Rotation about z-axis in degrees.
    
    Returns:
    ________________________________________________________________
    np.ndarray
        3x3 rotation matrix.
    """
    # Convert angles to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Compute rotation matrix
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

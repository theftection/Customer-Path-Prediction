import cv2
import numpy as np


def normalize_points(points, dim):
    """Normalize points to have zero mean and unit variance.
    Args:
        points (np.ndarray): Points to normalize.
        dim (int): Dimension of points.
    Returns:
        np.ndarray: Normalized points.
        np.ndarray: Transformation matrix.
    """
    centroid = np.mean(points, axis=0)
    centred_points = points - centroid
    scale = np.sqrt(dim) / np.mean(np.linalg.norm(centred_points, axis=1))
    T = np.diag([scale] * dim + [1])
    T[:dim, dim] = -scale * centroid
    print(T)
    print(T.dtype)
    return centred_points * scale, T


def estimate_projection_matrix(points_3D_norm, points_2D_norm):
    assert points_3D_norm.shape[0] == points_2D_norm.shape[0], "Number of points must be equal."
    assert points_3D_norm.shape[0] >= 6, "Number of points must be at least 6."
    n = points_2D_norm.shape[0]
    u = points_2D_norm[:,0]
    v = points_2D_norm[:,1]
    X = points_3D_norm[:,0]
    Y = points_3D_norm[:,1]
    Z = points_3D_norm[:,2]
    A = np.zeros((2*n, 12))

    for i in range(n):
        A[2*i,:] = [X[i], Y[i], Z[i], 1, 0, 0, 0, 0, -u[i]*X[i], -u[i]*Y[i], -u[i]*Z[i], -u[i]]
        A[2*i+1,:] = [0, 0, 0, 0, X[i], Y[i], Z[i], 1, -v[i]*X[i], -v[i]*Y[i], -v[i]*Z[i], -v[i]]

    U, S, V = np.linalg.svd(A)
    P = V[-1].reshape((3,4))

    return P

def compute_camera_center(P):
    Q = np.split(P, [3], axis=1)
    return np.squeeze(-np.matmul(np.linalg.inv(Q[0]), Q[1]))


def load_projection_matrix(project):
    P = np.load(f"inference_data/projection_matrix/{project}/projection_matrix.npy")
    P_inv = np.load(f"inference_data/projection_matrix/{project}/projection_matrix_inv.npy")
    camera_origin = np.load(f"inference_data/projection_matrix/{project}/camera_origin.npy")
    return P, P_inv, camera_origin



if __name__ == "__main__":

    # load points
    project = "Ch4_demo_960"
    points_3D = np.load(f"inference_data/projection_matrix/{project}/points/points_3D.npy")
    points_2D = np.load(f"inference_data/projection_matrix/{project}/points/points_2D.npy")

    # normalize points
    points_3D_norm, U = normalize_points(points_3D, 3)
    points_2D_norm, T = normalize_points(points_2D, 2)
    np.save(f"inference_data/projection_matrix/{project}/points/points_3D_norm.npy", points_3D_norm)
    np.save(f"inference_data/projection_matrix/{project}/points/points_2D_norm.npy", points_2D_norm)

    # estimate projection matrix
    P_norm = estimate_projection_matrix(points_3D_norm, points_2D_norm)
    P = np.linalg.inv(T) @ P_norm @ U
    camera_origin = compute_camera_center(P)

    # pseudo inverse
    P_inv_norm = np.linalg.pinv(P_norm)
    P_inv = np.linalg.inv(U) @ P_inv_norm @ T

    # save projection matrix
    np.save(f"inference_data/projection_matrix/{project}/projection_matrix.npy", P)
    np.save(f"inference_data/projection_matrix/{project}/projection_matrix_inv.npy", P_inv)
    np.save(f"inference_data/projection_matrix/{project}/camera_origin.npy", camera_origin)

import cv2
import numpy as np

from projection_matrix import load_projection_matrix


def project_2D_to_3D(P_inv, camera_origin, point_2D, height):
    """Project 2D point to 3D space.
    Args:
        P_inv (np.ndarray): Inverse of projection matrix.
        camera_origin (np.ndarray): Camera origin in 3D space.
        point_2D (np.ndarray): 2D point to project.
        height (float): Height of projected point.
    Returns:
        np.ndarray: 3D points.
    """
    point_2D_h = np.concatenate((point_2D, np.ones((point_2D.shape[0],1))), axis=1)
    point_3D_h = P_inv @ point_2D_h.T
    point_3D = (point_3D_h / point_3D_h[3])[:3].T
    ray_3D = point_3D - camera_origin
    t = (height - camera_origin[2]) / ray_3D[:, 2]
    world_point = camera_origin + ray_3D * t
    return world_point.astype(int)


def bb_2_map():
    ...

def bb_2_heigth():
    ...

def red_zone_estimation():
    ...


if __name__ == "__main__":
    # load points
    project = "test_impl"
    P, P_inv, camera_origin = load_projection_matrix(project)
    points_2D = np.array([[0, 0]])
    points_3D = project_2D_to_3D(P_inv, camera_origin, points_2D, 0)
    print(points_3D)







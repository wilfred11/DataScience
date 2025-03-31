import numpy as np


import numpy as np

def angle_between_vectors_np(u, v):
    u = np.array(u)
    v = np.array(v)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg

def vector_calc():
    vector_u = [3, 4,4]  # Vector u = (3, 4)
    vector_v = [5, -2,4]  # Vector v = (5, -2)
    angle_rad, angle_deg = angle_between_vectors_np(vector_u, vector_v)
    print(f"Angle between vectors (in radians): {angle_rad}")
    print(f"Angle between vectors (in degrees): {angle_deg}")

def vector():
    vector = np.array([3, 4])

    # Compute its magnitude manually
    length_manual = np.sqrt(vector[0] ** 2 + vector[1] ** 2)

    print("Vector length (manual calculation):", length_manual)

    length = np.linalg.norm(vector)
    print("Vector length:", length)  # Output: 5.0

    # Compute vector length using dot product
    length_dot = np.sqrt(np.dot(vector, vector))

    print("Vector length (dot product method):", length_dot)  # Output: 5.0

    # Define a 3D vector
    vector_3d = np.array([1, 2, 3])

    # Compute its length
    length_3d = np.linalg.norm(vector_3d)

    print("Vector length (3D):", length_3d)

    # Define a vector
    vector = np.array([3, -4])

    # Compute the L1 norm (Manhattan distance)
    length_l1 = np.linalg.norm(vector, ord=1)

    print("L1 norm:", length_l1)  # Output: 7
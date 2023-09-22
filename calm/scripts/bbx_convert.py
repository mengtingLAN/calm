import numpy as np

def convert_to_right_hand_Z_up(box_corners):
    transformation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    (x1, y1, z1), (x2, y2, z2) = box_corners

    def transform_point(x, y, z):
        point = np.array([x, y, z])
        return tuple(np.dot(transformation_matrix, point))


    corners = [
        transform_point(x1, y1, z1),
        transform_point(x2, y1, z1),
        transform_point(x2, y2, z1),
        transform_point(x1, y2, z1),
        transform_point(x1, y1, z2),
        transform_point(x2, y1, z2),
        transform_point(x2, y2, z2),
        transform_point(x1, y2, z2),
    ]

    corners_np = np.array(corners)

    np.save("./bbx.npy",corners_np)

    return corners


box_corners = [(-0.223+0.001, -0.000-0.382, -0.254+0.013), (0.221+0.001, 0.764-0.382, 0.229+0.013)]
print("Box corners in right-hand coordinate system:")
for i, corner in enumerate(convert_to_right_hand_Z_up(box_corners)):
    print(f"P{i+1}: {corner}")


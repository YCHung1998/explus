import numpy as np
import math


def auto_image_collage_with_padding(image_list: list) -> np.ndarray:
    """Combines N images into a k x k collage with black padding.

    Args:
        image_list: List of NumPy arrays of shape (H, W) or (H, W, C).

    Returns:
        A single NumPy array representing the k x k collage.

    Raises:
        ValueError: If the input list is empty.
    """
    if not image_list:
        raise ValueError("Input image list cannot be empty.")

    num_images = len(image_list)
    grid_size = math.ceil(math.sqrt(num_images))
    total_slots = grid_size * grid_size

    # Prepare padding
    img_shape = image_list[0].shape
    img_dtype = image_list[0].dtype
    padding_count = total_slots - num_images

    black_img = np.zeros(img_shape, dtype=img_dtype)
    full_list = image_list + [black_img] * padding_count

    # Build collage using horizontal and vertical stacking
    rows = []
    for i in range(grid_size):
        row_start = i * grid_size
        row_end = (i + 1) * grid_size
        row_images = full_list[row_start:row_end]
        rows.append(np.hstack(row_images))

    return np.vstack(rows)

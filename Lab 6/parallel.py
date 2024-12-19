def parallel_cellular_edge_detection(image, iterations=1, threshold=30):
    """
    Edge detection using Parallel Cellular Algorithm.

    Parameters:
        image (2D numpy array): Grayscale image.
        iterations (int): Number of iterations to process.
        threshold (int): Intensity difference threshold for edge detection.

    Returns:
        2D numpy array: Binary edge-detected image.
    """
    import numpy as np

    # Ensure the input is a valid 2D NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid 2D NumPy array (grayscale image).")
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image with shape (rows, cols).")

    # Get image dimensions
    rows, cols = image.shape

    # Define neighbor offsets (Moore neighborhood: 8 neighbors)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    # Initialize the edge map
    edge_map = np.zeros_like(image)

    for _ in range(iterations):
        # Create a copy of the current image
        new_image = image.copy()

        # Iterate over each pixel (excluding borders)
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                # Check intensity differences with neighbors
                for dx, dy in neighbors:
                    neighbor_x, neighbor_y = x + dx, y + dy
                    if abs(image[x, y] - image[neighbor_x, neighbor_y]) > threshold:
                        edge_map[x, y] = 255  # Mark as edge
                        break
                else:
                    edge_map[x, y] = 0  # Not an edge

        # Update the image for the next iteration
        image = new_image

    return edge_map

# Example Usage
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Sample data representing a grayscale image (replace with your data)
    image_data = np.array([
        [100, 100, 100, 100, 100],
        [100, 200, 200, 200, 100],
        [100, 200, 10 , 200, 100],
        [100, 200, 200, 200, 100],
        [100, 100, 100, 100, 100],
    ], dtype=np.uint8)  # Adjust data type as needed

    # Apply Parallel Cellular Algorithm for edge detection
    edges = parallel_cellular_edge_detection(image_data, iterations=1, threshold=30)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_data, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Edge Detection Result")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

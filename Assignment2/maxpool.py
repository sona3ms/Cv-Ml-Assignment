import numpy as np

# Define the matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# Define the window size
window_size = 2  # Example window size of 2x2

# Define the max_pooling_basic function
def max_pooling_basic(image, window_size):
    m, n = image.shape
    k = window_size
    
    result = np.zeros((m - k + 1, n - k + 1))
    
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            result[i, j] = np.max(image[i:i+k, j:j+k])
    
    return result

# Call the max_pooling_basic function
result = max_pooling_basic(matrix, window_size)

# Print the result
print("Result of max pooling:")
print(result)

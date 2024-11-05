import numpy as np
import time
import matplotlib.pyplot as plt

def histogram_plot(img, bins=256, hist_range=[0, 256]):
    """
    example usage:

    upper_bound = int(np.percentile(img, 99))
    img = np.clip(img, 0, upper_bound)

    cdf = histogram_plot(img, bins=upper_bound + 1, hist_range=[0, upper_bound + 1])
    eqed_cdf = compute_equalized_cdf(cdf, upper_bound, 'uint16')
    eqed_img = eqed_cdf[img]
    histogram_plot(eqed_img)
    """
    hist, bins = np.histogram(img.flatten(), bins, hist_range)
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Plot with log scale for y-axis
    plt.figure()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), bins, hist_range, color='r')
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()



def get_start_point_list(img):
    """
    start from four corners
    """
    start_list = []
    for i in range(img.shape[1]):
        start_list.append((0, i))
        start_list.append((img.shape[0] - 1, i))
    for i in range(img.shape[0]):
        start_list.append((i, 0))
        start_list.append((i, img.shape[1] - 1))
    return start_list


def get_neibor(cur, img):
    displacement = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    neibor_list = []
    for delta in displacement:
        if 0 <= cur[0] + delta[0] < img.shape[0] \
                and 0 <= cur[1] + delta[1] < img.shape[1]:
            neibor_list.append((cur[0] + delta[0], cur[1] + delta[1]))
    return neibor_list

def is_in_brain(cur, img,thres):
    if img[cur[0], cur[1]] >= thres:
        return True
    else:
        return False


def bfs(img,thres):
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)  # Squeeze out the first dimension
    queue = []
    visited = np.zeros(shape=img.shape, dtype=np.bool)
    start_list = get_start_point_list(img)
    for st in start_list:
        queue.append(st)
    while len(queue) > 0:
        cur = queue[0]
        queue.pop(0)
        if visited[cur[0], cur[1]]:
            continue
        if is_in_brain(cur, img,thres):
            continue
        visited[cur[0], cur[1]] = True
        neighbor_list = get_neibor(cur, img)
        for neighbor in neighbor_list:
            queue.append(neighbor)
    visited = 1 - visited
    return visited



def time_execution(func):
 
    """ 
     Example usage
     @time_execution
     def example_function():
         Simulate some task
         time.sleep(1)
     example_function()  
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate the time difference
        print(f"Execution time: {execution_time:.4f} seconds")
        return result
    return wrapper






        
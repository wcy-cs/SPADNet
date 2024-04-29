import numpy as np

def generate_gt(size, landmark_list, sigma=1):
    '''
    return N * H * W
    '''
    heatmap_list = [
        _generate_one_heatmap(size, l, sigma) for l in landmark_list
    ]
    return np.stack(heatmap_list, axis=0)

def _generate_one_heatmap(size, landmark, sigma):
    w = size
    h = size
    # print("landmark: ", landmark.shape)
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    # print(x_range.shape)
    xx, yy = np.meshgrid(x_range, y_range)
    # print("xx: ", xx.shape, "yy: ", yy.shape)
    d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    # print(type(heatmap[0][0]))
    return heatmap


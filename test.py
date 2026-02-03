import numpy as np
import matplotlib.pyplot as plt
x_range = np.arange(-0.3, 3.0, 0.2)
print(x_range.shape)

y_range = np.arange(-1.45, 0.25, 0.2)
print(y_range.shape)
def generate_grid(x_range, y_range):
    all_points = []
    column_id = 0
    for y in y_range:
        if column_id%2 == 0:
            x_range_used = x_range
        else:
            x_range_used = x_range[::-1]
        for x in x_range_used:
            all_points.append((x, y))
        column_id += 1
    return np.array(all_points)
print(generate_grid(x_range, y_range))
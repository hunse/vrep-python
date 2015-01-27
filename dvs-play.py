"""
Play DVS events in real time

TODO: deal with looping event times for recordings > 65 s
"""

import time

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# events = np.load('dvs-epuck-27s.npy')
events = np.load('dvs-ball-2s.npy')
ts = events['t'] / 1000.
assert (np.diff(ts) >= 0).all()

t_world = time.time()
t_max = ts[-1]
t0 = 0

plt.figure(1)
plt.clf()
image = np.zeros((128, 128), dtype=float)
plt_image = plt.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation=None)
plt.gca().invert_yaxis()

while t0 < t_max:
    time.sleep(0.001)

    t1 = time.time() - t_world

    new_events = events[(ts > t0) & (ts < t1)]

    dt = t1 - t0
    image *= np.exp(-dt / 0.01)

    for x, y, s, _ in new_events:
        image[y, x] += 1 if s else -1

    plt_image.set_data(image)
    plt.draw()

    t0 = t1

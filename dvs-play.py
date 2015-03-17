"""
Play DVS events in real time

TODO: deal with looping event times for recordings > 65 s
"""
import numpy as np
import matplotlib.pyplot as plt

import dvs

def close(a, b, atol=1e-8, rtol=1e-5):
    return np.abs(a - b) < atol + rtol * b

def imshow(image, ax=None):
    ax = plt.gca() if ax is None else ax
    ax.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation=None)

def add_to_image(image, events):
    for x, y, s, _ in events:
        image[y, x] += 1 if s else -1

def as_image(events):
    image = np.zeros((128, 128), dtype=float)
    add_to_image(image, events)
    return image


# filename = 'dvs.npz'
filename = 'dvs-ball-10ms.npz'
events = dvs.load(filename, dt_round=True)

udiffs = np.unique(np.diff(np.unique(events['t'])))
assert np.allclose(udiffs, 0.01)

plt.figure(1)
plt.clf()
times = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for i in range(6):
    plt.subplot(2, 3, i+1)
    imshow(as_image(events[close(events['t'], times[i])]))
    plt.title("t = %0.3f" % times[i])


# plt.figure(1)
# plt.clf()
# image = np.zeros((128, 128), dtype=float)
# plt_image = plt.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation=None)
# plt.gca().invert_yaxis()

# while t0 < t_max:
#     time.sleep(0.001)

#     t1 = time.time() - t_world

#     new_events = events[(ts > t0) & (ts < t1)]

#     dt = t1 - t0
#     image *= np.exp(-dt / 0.01)

#     for x, y, s, _ in new_events:
#         image[y, x] += 1 if s else -1

#     plt_image.set_data(image)
#     plt.draw()

#     t0 = t1

plt.show()

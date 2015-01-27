
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
plt.ion()

DEBUG = False

events = np.load('dvs-epuck-27s.npy')
# events = np.load('dvs-ball-2s.npy')
ts = events['t'] / 1000.
assert (np.diff(ts) >= 0).all()

dt = 1e-3
t0 = 6.0
t1 = t0 + 1.5
nt = int((t1 - t0) / dt) + 1
# nt = min(nt, 1000)  # cap at 1000 for now

image = np.zeros((128, 128))
images = np.zeros((nt, 128, 128))

for i in range(nt):
    # --- decay image
    image *= np.exp(-dt / 0.01)
    # image *= 0

    # --- add events
    ti = t0 + i * dt
    eventsi = events[ts == ti]

    for x, y, s, _ in eventsi:
        image[y, x] += 1 if s else -1

    images[i] = image

# --- average in frames
dt_frame = 0.01
nt_frame = int(dt_frame / dt)
nt_video = int(nt / nt_frame)

video_image = np.zeros((nt_video, 128, 128))
for i in range(nt_video):
    slicei = slice(i*nt_frame, (i+1)*nt_frame)
    video_image[i] = np.sum(images[slicei], axis=0)

# --- play video
fig = plt.figure(1)
fig.clf()
axes = [plt.gca()]

plt_image = axes[0].imshow(video_image[0], vmin=-1, vmax=1, cmap='gray', interpolation=None)
axes[0].invert_yaxis()

def update(i, video_image, axes, plt_image):
    plt_image.set_data(video_image[i])
    axes[0].set_title("t = %0.3f" % ((i + 1) * dt_frame))
    return plt_image,

ani = matplotlib.animation.FuncAnimation(
    fig, update, nt_video,
    fargs=(video_image, axes, plt_image),
    interval=10, blit=False)

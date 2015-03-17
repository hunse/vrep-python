import collections
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
plt.ion()

import dvs

filename = 'dvs-ball-1ms.npz'
events = dvs.load(filename, dt_round=True)

dvs.flow(events, debug=True)
assert False

def imshow(image, ax=None):
    ax = plt.gca() if ax is None else ax
    # ax.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation=None)
    ax.imshow(image, cmap='gray', interpolation=None)

def gaussian(x, mean=0, std=1, normalize=True):
    y = np.exp(-(0.5 / std**2) * (x - mean)**2)
    y /= y.sum()
    return y

# def convolve_fft(x, y, axis=-1):
#     X = np.fft.fft(x, axis=axis)
#     Y

DEBUG = True
# DEBUG = False
debug_fig = plt.figure(1)
debug_fig.clf()

events = np.load('dvs-epuck-27s.npy')
# events = np.load('dvs-ball-2s.npy')
ts = events['t'] / 1000.
assert (np.diff(ts) >= 0).all()

# --- compute flow
dt = 1e-3
t0 = 6.0
# t0 = 0.0
t1 = t0 + 1
nt = int((t1 - t0) / dt) + 1
# nt = min(nt, 1000)  # cap at 1000 for now

df = 5
nf = int(np.ceil(128. / df))

images = np.zeros((nt, 128, 128))
flows = np.nan * np.ones((nt, nf, nf, 2))

n_grids = 20
grids = collections.deque(maxlen=n_grids)

for it in range(nt):
    ti = t0 + it * dt
    eventsi = events[ts == ti]

    # --- make image
    for x, y, s, _ in eventsi:
        images[it, y, x] += 1 if s else -1

    # --- compute flows
    grid = np.zeros((df * nf, df * nf))
    for x, y, s, _ in eventsi:
        grid[y, x] += 1 if s else -1
    grids.append(np.array(grid))

    # sum grids
    for g in list(grids)[:-1]:
        grid += g

    if 0:
        dx = np.array(grid)
        dx[:, 1:] -= dx[:, :-1]
        dy = np.array(grid)
        # dy[1:, :] -= dy[:-1, :]
        dy[1:, :] = np.diff(dy, axis=0)
    else:
        # calculate derivative on smoothed image
        # TODO: be more efficient
        import cv2
        # grid_s = cv2.GaussianBlur(grid, (9, 9), 1)
        grid_s = cv2.GaussianBlur(grid, (9, 9), 3)
        dx = np.array(grid_s)
        dx[:, 1:] -= dx[:, :-1]
        dy = np.array(grid_s)
        dy[1:, :] = np.diff(dy, axis=0)

    dT = grid / (dt * n_grids)

    dx.shape = (nf, df, nf, df)
    dy.shape = (nf, df, nf, df)
    dT.shape = (nf, df, nf, df)

    dXY = np.zeros((df * df, 2))
    for y in range(nf):
        for x in range(nf):
            dXY[:, 0] = dx[y, :, x, :].ravel()
            dXY[:, 1] = dy[y, :, x, :].ravel()
            v, _, _, _ = np.linalg.lstsq(dXY, dT[y, :, x, :].ravel())
            flows[it, y, x, :] = v

    print("Frame %d (%d events)" % (it, len(eventsi)))
    if 'DEBUG' in globals() and DEBUG and it > 100:
        debug_fig.clf()
        axes = [debug_fig.add_subplot(2, 2, k+1) for k in range(4)]
        imshow(grid, ax=axes[0])
        # axes[1].quiver(flows[it, :, :, 0], flows[it, :, :, 1])
        flow = flows[slice(max(it - 50, 0), it)].mean(axis=0)
        axes[1].quiver(flow[:, :, 0], flow[:, :, 1])
        imshow(dx.reshape(nf*df, nf*df), ax=axes[2])
        imshow(dy.reshape(nf*df, nf*df), ax=axes[3])
        (ax.invert_yaxis() for ax in [axes[0], axes[2], axes[3]])

        plt.draw()
        raw_input("Press any key...")

# --- average in frames
dt_frame = 0.01
nt_frame = int(dt_frame / dt)
nt_video = int(nt / nt_frame)

video_image = np.zeros((nt_video, 128, 128))
video_flow = np.zeros((nt_video, nf, nf, 2))
for i in range(nt_video):
    slicei = slice(i*nt_frame, (i+1)*nt_frame)
    video_image[i] = np.sum(images[slicei], axis=0)
    video_flow[i] = np.nanmean(flows[slicei], axis=0)

# --- play video
fig = plt.figure(1)
fig.clf()
r, c = 2, 1
axes = [fig.add_subplot(r, c, i + 1) for i in range(r * c)]

plt_image = axes[0].imshow(video_image[0], vmin=-1, vmax=1, cmap='gray', interpolation=None)
axes[0].invert_yaxis()

plt_quiver = axes[1].quiver(video_flow[0, :, :, 0], video_flow[0, :, :, 1])

def quiver_update(i, video_image, video_flow, axes, plt_image, plt_quiver):
    plt_image.set_data(video_image[i])
    plt_quiver.set_UVC(video_flow[i, :, :, 0], video_flow[i, :, :, 1])
    axes[0].set_title("t = %0.3f" % ((i + 1) * dt_frame))
    return plt_image, plt_quiver

ani = matplotlib.animation.FuncAnimation(
    fig, quiver_update, nt_video,
    fargs=(video_image, video_flow, axes, plt_image, plt_quiver),
    interval=100, blit=False)

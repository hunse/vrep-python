"""Try representing the spiking data with Gabor RFs.

1. Give an ensemble a bunch of Gabor encoders
"""
import numpy as np
import matplotlib.pyplot as plt
import nengo
import vanhateren
plt.ion()

import dvs
import gabor

from hunse_tools.timing import tic, toc

def show_image(ax, image):
    plt_img = ax.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    return plt_img

rng = np.random.RandomState(3)
dt = 0.001

# --- load and filter DVS spikes
filename = 'dvs.npz'
# filename = 'dvs-ball-1ms.npz'
events = dvs.load(filename, dt_round=False)

t0 = 0.1
spikes = dvs.make_video(events, t0=t0, dt_frame=dt, tau=0)
video = dvs.make_video(events, t0=t0, dt_frame=dt, tau=0.005)

if 0:
    plt.ion()
    plt.figure(1)
    ax = plt.gca()
    img = show_image(ax, video[0])
    ax.invert_yaxis()

    for frame in video:
        img.set_data(frame)
        plt.draw()
        raw_input("Pause...")

if 0:
    plt.figure(1)
    ani = dvs.animate([video], [plt.gca()])

    plt.show()

# --- generate a filterbank

n = 5000
# rf_shape = (9, 9)
rf_shape = (15, 15)
thetas = rng.uniform(0, np.pi, size=n)
# phases = rng.uniform(-0.5, 0.5, size=n)
freqs = rng.uniform(0.1, 2.0, size=n)
# freqs = rng.uniform(0.2, 0.7, size=n)

# data_set = gabors(thetas, phases=0, freqs=0.5, sigmas_y=10., shape=rf_shape)
bank = gabor.bank(thetas, freqs=freqs, sigmas_y=10., shape=rf_shape)

if 0:
    # test Gabors
    plt.figure(1)
    plt.clf()

    r, c = 5, 5
    for i in range(min(len(bank), r * c)):
        plt.subplot(r, c, i+1)
        plt.imshow(bank[i].reshape(rf_shape), cmap='gray', interpolation='none')
        plt.xticks([])

    # pair = filter_pair((15, 15), 1.0, 0.5, 1.0)
    # axs = [plt.subplot(1, 2, i+1) for i in range(2)]
    # axs[0].imshow(pair.real)
    # axs[1].imshow(pair.imag)

    plt.show()
    assert False

# --- make encoders
im_shape = (128, 128)
inds = gabor.inds(len(bank), im_shape, rf_shape, rng=rng)
encoders = gabor.matrix_from_inds(bank, inds, im_shape)

# --- build network and solve for decoders
im_dims = np.prod(im_shape)

n_eval_points = 5000
if 0:
    # Van Hateren
    patches = vanhateren.VanHateren().patches(n_eval_points, im_shape)
    patches = vanhateren.preprocess.scale(patches)
    patches.shape = (-1, im_dims)
    eval_points = patches.reshape(-1, im_dims)
elif 0:
    # More gabors
    rf_shape = (32, 32)
    thetas = rng.uniform(0, np.pi, size=n_eval_points)
    # phases = rng.uniform(-0.5, 0.5, size=n)
    freqs = rng.uniform(0.1, 2, size=n_eval_points)

    eval_bank = gabor.bank(thetas, freqs=freqs, sigmas_y=10., shape=rf_shape)
    inds = gabor.inds(len(eval_bank), im_shape, rf_shape, rng=rng)
    eval_points = gabor.matrix_from_inds(eval_bank, inds, im_shape)
else:
    # Low freq images
    cutoff = 0.1
    images = rng.normal(size=(n_eval_points,) + im_shape)
    X = np.fft.fft2(images)
    f0 = np.fft.fftfreq(X.shape[-2])
    f1 = np.fft.fftfreq(X.shape[-1])
    ff = np.sqrt(f0[:, None]**2 + f1[None, :]**2)
    X[..., ff > cutoff] = 0
    Y = np.fft.ifft2(X)
    assert np.allclose(Y.imag, 0)
    eval_points = Y.real.clip(-1, 1).reshape(-1, im_dims)

fig_uv = plt.figure(2)
plt.clf()
plt.subplot(211)
u_plot = plt.imshow(video[0], vmin=-1, vmax=1, cmap='gray')
plt.subplot(212)
v_plot = plt.imshow(video[0], vmin=-1, vmax=1, cmap='gray')

def video_input(t):
    i = int(np.round(t / dt))
    return video[i % len(video)].ravel()

def video_plot(t, y):
    x = video_input(t).reshape(im_shape)
    y = y.reshape(im_shape)
    u_plot.set_data(x)
    v_plot.set_data(y)
    fig_uv.canvas.draw()

rmse_total = np.array(0.0)
def rmse_recorder(t, y):
    x = video_input(t)
    if t > 0:
        rmse_total[...] += np.sqrt(((y - x)**2).mean())

net = nengo.Network(seed=9)
with net:
    u = nengo.Node(video_input)
    # a = nengo.Ensemble(n, im_dims, encoders=encoders)
    a = nengo.Ensemble(n, im_dims, encoders=encoders, intercepts=nengo.dists.Choice([0]))
    nengo.Connection(u, a)

    # a = nengo.Ensemble(n, im_dims)
    v = nengo.Node(size_in=im_dims, size_out=im_dims)
    c = nengo.Connection(a, v, eval_points=eval_points, synapse=0.01)

    w1 = nengo.Node(video_plot, size_in=im_dims)
    w2 = nengo.Node(rmse_recorder, size_in=im_dims)
    nengo.Connection(v, w1, synapse=None)
    nengo.Connection(v, w2, synapse=None)

sim = nengo.Simulator(net)

# show reconstructed images
if 1:
    from nengo.builder.connection import build_linear_system

    plt.figure(3)
    plt.clf()

    _, A, y = build_linear_system(sim.model, c, None)
    x = sim.data[c].decoders.T

    r, c = 2, 5
    axes = [[plt.subplot2grid((2*r, c), (2*i+k, j)) for k in range(2)]
            for i in range(r) for j in range(c)]
    for i in range(r * c):
        show_image(axes[i][0], y[i].reshape(im_shape))
        show_image(axes[i][1], np.dot(A[i], x).reshape(im_shape))

sim.run(1.)
print(rmse_total)

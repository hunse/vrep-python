import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import vrep
oneshot = vrep.simx_opmode_oneshot


def _process_events(cid, attr, mode=oneshot, last=None):
    err, data = vrep.simxGetStringSignal(cid, attr, mode)

    x, y, signs, times = [], [], [], []
    if err or len(data) == 0:
        pass
    elif last is not None and (len(data) == last[0] and data[:4] == last[1]):
        pass
    else:
        last = (len(data), data[:4])

        # --- Format data
        data = np.array(bytearray(data), dtype=np.uint8)
        data.shape = (-1, 4)

        signs = data[:, 0] >= 128
        data[:, 0] -= signs * 128

        x, y = data[:, 0], data[:, 1]
        times = 256 * data[:, 3] + data[:, 2]

    return (x, y, signs, times), last


def _combine_events(events):
    n = sum(len(x) for x, y, s, t in events)
    record = np.zeros(
        n, dtype=[('x', 'u1'), ('y', 'u1'), ('s', 'b'), ('t', 'u2')])

    i = 0
    for x, y, s, t in events:
        n = len(x)
        record[i:i + n]['x'] = x
        record[i:i + n]['y'] = y
        record[i:i + n]['s'] = s
        record[i:i + n]['t'] = t
        i += n

    return record


def record(filename='dvs.npz', stereo=False):

    cid = vrep.simxStart(
        connectionAddress='127.0.0.1',
        connectionPort=19997,
        waitUntilConnected=True,
        doNotReconnectOnceDisconnected=True,
        timeOutInMs=5000,
        commThreadCycleInMs=5)

    if cid != -1:
        print("Connected to V-REP remote API server, client id: %s" % cid)
    else:
        raise RuntimeError("Failed connecting to V-REP remote API server")


    vrep.simxStartSimulation(cid, oneshot)

    try:
        events0 = []
        events1 = []

        last0 = (-1, '')
        last1 = (-1, '')

        while True:
            e0, last0 = _process_events(cid, "currentDVS", oneshot, last0)
            if stereo:
                e1, last1 = _process_events(cid, "currentDVS0", oneshot, last1)

            # --- Store data
            if len(e0[0]) > 0:
                events0.append(e0)
            if stereo and len(e1[0]) > 0:
                events1.append(e1)

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        # --- Stop simulation
        while vrep.simxStopSimulation(cid, oneshot):
            time.sleep(0.01)
        vrep.simxFinish(cid)
        print("Simulation stopped")

    # --- Save data
    events0 = _combine_events(events0)
    if stereo:
        events1 = _combine_events(events1)

    if filename is not None:
        if not stereo:
            np.savez(filename, events0=events0)
        else:
            np.savez(filename, events0=events0, events1=events1)
        print("Saved '%s'" % filename)

    return (events0, events1) if stereo else events0


def _load_events(record):
    events = np.zeros(len(record),
        dtype=[('x', 'u1'), ('y', 'u1'), ('s', 'b'), ('t', 'f4')])

    events['x'] = record['x']
    events['y'] = record['y']
    events['s'] = record['s']
    events['t'] = record['t'] / 1000.
    assert (np.diff(events['t']) >= 0).all()

    return events


def load(filename='dvs.npz'):
    data = np.load(filename)
    events0 = _load_events(data['events0'])
    if 'events1' in data:
        events1 = _load_events(data['events1'])
        return events0, events1
    else:
        return events0


def make_video(events, t0=0.0, t1=None, dt_frame=0.01):
    if t1 is None:
        t1 = events['t'].max()

    ts = events['t']
    dt = 1e-3
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
    nt_frame = int(dt_frame / dt)
    nt_video = int(nt / nt_frame)

    video = np.zeros((nt_video, 128, 128))
    for i in range(nt_video):
        slicei = slice(i*nt_frame, (i+1)*nt_frame)
        video[i] = np.sum(images[slicei], axis=0)

    return video


def show(events, t0=0.0, t1=None, axs=None, stereo=False):
    import matplotlib.animation

    if stereo:
        assert axs is not None and len(axs) == 2
        assert len(events) == 2
    else:
        if axs is None:
            axs = plt.gca()
        axs = [axs]
        events = [events]

    if stereo and t1 is None:
        t1 = max(events[0]['t'].max(), events[1]['t'].max())

    dt_frame = 0.01
    videos = []
    for eventsi in events:
        videos.append(make_video(eventsi, t0, t1, dt_frame=dt_frame))

    assert all(len(video) == len(videos[0]) for video in videos)

    plt_images = []
    for ax, video in zip(axs, videos):
        plt_images.append(ax.imshow(video[0], vmin=-1, vmax=1, cmap='gray', interpolation=None))
        ax.invert_yaxis()

    def update(i, videos, axs, plt_images):
        for ax, video, plt_image in zip(axs, videos, plt_images):
            plt_image.set_data(video[i])
            ax.set_title("t = %0.3f" % (t0 + (i + 1) * dt_frame))
        return plt_images,

    ani = matplotlib.animation.FuncAnimation(
        ax.figure, update, len(videos[0]),
        fargs=(videos, axs, plt_images),
        interval=10, blit=False)

    return ani


def test_record_stereo():
    filename = 'dvs-epuck-stereo.npz'
    # record(filename, stereo=True)
    events0, events1 = load(filename)

    plt.ion()
    plt.figure(101)
    plt.clf()
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)
    ani = show([events0, events1], axs=[ax0, ax1], t0=10, t1=11, stereo=True)
    return ani


if __name__ == '__main__':
    ani = test_record_stereo()

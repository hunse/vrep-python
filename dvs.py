import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import vrep
oneshot = vrep.simx_opmode_oneshot
oneshot_wait = vrep.simx_opmode_oneshot_wait


def close(a, b, atol=1e-8, rtol=1e-5):
    return np.abs(a - b) < atol + rtol * b


def imshow(image, ax=None):
    ax = plt.gca() if ax is None else ax
    ax.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation=None)
    ax.invert_yaxis()


def add_to_image(image, events):
    for x, y, s, _ in events:
        image[y, x] += 1 if s else -1
    return image


def as_image(events):
    return add_to_image(np.zeros((128, 128)), events)


def error_messages(error_code):
    error_pairs = [
        (vrep.simx_error_novalue_flag, 'novalue'),
        (vrep.simx_error_timeout_flag, 'timeout'),
        (vrep.simx_error_illegal_opmode_flag, 'illegal opmode'),
        (vrep.simx_error_remote_error_flag, 'remote error'),
        (vrep.simx_error_split_progress_flag, 'split progress'),
        (vrep.simx_error_local_error_flag, 'local error'),
        (vrep.simx_error_initialize_error_flag, 'initialize error')
    ]

    errors = [v for k, v in error_pairs if error_code % (2 * k) >= k]
    return errors


def safe_call(vrep_fn, *args, **kwargs):
    n_tries = kwargs.pop('n_tries', 10)
    novalue_error = kwargs.pop('novalue_error', False)

    for _ in range(n_tries):
        output = vrep_fn(*args, **kwargs)
        if isinstance(output, int):
            # must be an error code
            err = output
            output = None
        else:
            err, output = output

        if err == vrep.simx_error_noerror or (
                novalue_error is False and err == vrep.simx_error_novalue_flag):
            return output
        elif err == vrep.simx_error_timeout_flag or (
                novalue_error is None and err == vrep.simx_error_novalue_flag):
            time.sleep(0.01)
            continue
        else:
            err_string = ', '.join(error_messages(err))
            raise RuntimeError("Could not execute '%s' due to errors:\n  %s"
                               % (vrep_fn.__name__, err_string))

    err_string = ', '.join(error_messages(err))
    raise RuntimeError("Could not execute '%s' after %d tries due to:\n  %s" %
                       (vrep_fn.__name__, n_tries, err_string))


def _process_events(cid, attr, mode=oneshot_wait, last=None):
    err, data = vrep.simxGetStringSignal(cid, attr, mode)
    safe_call(vrep.simxClearStringSignal, cid, attr, mode)

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


def record(filename='dvs.npz', stereo=False, continuous=False):

    cid = vrep.simxStart(
        connectionAddress='127.0.0.1',
        connectionPort=19997,
        waitUntilConnected=True,
        doNotReconnectOnceDisconnected=True,
        timeOutInMs=5000,
        commThreadCycleInMs=5)

    if cid != -1:
        print("Connected to V-REP remote API server, client id: %s" % cid)
        time.sleep(0.01)
    else:
        raise RuntimeError("Failed connecting to V-REP remote API server")


    dt = safe_call(vrep.simxGetFloatingParameter,
                   cid, vrep.sim_floatparam_simulation_time_step, oneshot_wait)
    safe_call(vrep.simxStartSimulation, cid, oneshot_wait, novalue_error=None)

    try:
        events0 = []
        events1 = []

        if continuous:
            # Record continuously (does not work well)
            last0 = (-1, '')
            last1 = (-1, '')

            try:
                while True:
                    e0, last0 = _process_events(cid, "currentDVS", oneshot_wait, last0)
                    if stereo:
                        e1, last1 = _process_events(cid, "currentDVS0", oneshot_wait, last1)

                    # --- Store data
                    if len(e0[0]) > 0:
                        events0.append(e0)
                    if stereo and len(e1[0]) > 0:
                        events1.append(e1)

                    time.sleep(0.001)

            except KeyboardInterrupt:
                pass

        else:
            # Wait until interrupt to record
            try:
                while True:
                    time.sleep(0.01)
            except KeyboardInterrupt:
                pass

            e0, _ = _process_events(cid, "currentDVS", oneshot_wait)
            if stereo:
                e1, _ = _process_events(cid, "currentDVS0", oneshot_wait)

            # --- Store data
            events0.append(e0)
            if stereo:
                events1.append(e1)

    finally:
        # --- Stop simulation
        safe_call(vrep.simxStopSimulation, cid, oneshot_wait, novalue_error=None)
        vrep.simxFinish(cid)
        print("Simulation stopped")

    # --- Save data
    events0 = _combine_events(events0)
    if stereo:
        events1 = _combine_events(events1)

    if filename is not None:
        if not stereo:
            np.savez(filename, events0=events0, dt=dt)
        else:
            np.savez(filename, events0=events0, events1=events1, dt=dt)
        print("Saved '%s'" % filename)

    return (events0, events1) if stereo else events0


def _load_events(raw_events, dt=None):
    events = np.zeros(len(raw_events),
        dtype=[('x', 'u1'), ('y', 'u1'), ('s', 'b'), ('t', 'f8')])

    events['x'] = raw_events['x']
    events['y'] = raw_events['y']
    events['s'] = raw_events['s']
    if dt is not None:
        dtr = int(np.round(dt / 1e-4)) * 1e-4  # round dt to nearest 100 us
        t = np.round(raw_events['t'] / (1000. * dt))
        events['t'] = t * dtr
    else:
        events['t'] = raw_events['t'] / 1000.

    assert (np.diff(events['t']) >= 0).all()

    return events


def load(filename='dvs.npz', dt_round=False):
    data = np.load(filename)

    dt = data['dt'] if dt_round else None
    events0 = _load_events(data['events0'], dt=dt)
    if 'events1' in data:
        events1 = _load_events(data['events1'], dt=dt)
        return events0, events1
    else:
        return events0


def make_video(events, t0=0.0, t1=None, dt_frame=0.01, tau=0.01):
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
        image *= np.exp(-dt / tau) if tau > 0 else 0
        # image *= 0

        # --- add events
        ti = t0 + i * dt
        add_to_image(image, events[close(ts, ti)])

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


def animate(videos, axes):
    import matplotlib.animation

    plt_images = []
    for ax, video in zip(axes, videos):
        plt_images.append(ax.imshow(video[0], vmin=-1, vmax=1, cmap='gray', interpolation=None))
        ax.invert_yaxis()

    def update(i, videos, axes, plt_images):
        for ax, video, plt_image in zip(axes, videos, plt_images):
            plt_image.set_data(video[i])
            ax.set_title("i = %d" % i)
        return plt_images,

    ani = matplotlib.animation.FuncAnimation(
        ax.figure, update, len(videos[0]),
        fargs=(videos, axes, plt_images),
        interval=10, blit=False)

    return ani


def flow(events, debug=False):
    import collections

    if debug:
        debug_fig = plt.figure(19)
        debug_fig.clf()

    dt = np.unique(np.diff(np.unique(events['t']))).min()
    print(dt)

    t0 = 0.4
    t1 = events['t'].max()
    nt = int((t1 - t0) / dt) + 1

    df = 5
    nf = int(np.ceil(128. / df))

    images = np.zeros((nt, 128, 128))
    flows = np.nan * np.ones((nt, nf, nf, 2))

    n_grids = 10
    grids = collections.deque(maxlen=n_grids)

    for it in range(nt):
        ti = t0 + it * dt
        eventsi = events[close(events['t'], ti)]

        # --- make image
        add_to_image(images[it], eventsi)

        # --- compute flows
        grid = np.zeros((df * nf, df * nf))
        add_to_image(grid, eventsi)
        grids.append(np.array(grid))

        # sum grids
        for g in list(grids)[:-1]:
            grid += g

        if 0:
            dx = np.array(grid)
            dx[:, 1:] -= dx[:, :-1]
            dy = np.array(grid)
            dy[1:, :] = np.diff(dy, axis=0)
        else:
            # calculate derivative on smoothed image
            # TODO: be more efficient

            import cv2
            grid_s = cv2.GaussianBlur(grid, (9, 9), 3)
            dx = np.array(grid_s)
            dx[:, 1:] -= dx[:, :-1]
            dy = np.array(grid_s)
            dy[1:, :] = np.diff(dy, axis=0)


        # if len(grids) > 1:
        #     dT = (grids[-1] - grids[-2]) / dt
        # else:
        #     dT = grid / dt

        # dT = grid / (dt * n_grids)
        # dT = grid / dt
        # dT = np.array(grid)
        dT = np.array(grids[-1])

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

        flow = flows[it]
        flow[abs(flow) > 1e3] = 0

        print("Frame %d (%d events)" % (it, len(eventsi)))
        print(abs(dx).max(), abs(dy).max(), abs(dT).max())
        if debug and it > 50 and it % 10 == 0:
            debug_fig.clf()
            gs = gridspec.GridSpec(1, 2)
            gs0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0])
            axes = [debug_fig.add_subplot(gs0[k]) for k in range(4)]
            imshow(grid, ax=axes[0])
            imshow(dT.reshape(nf*df, nf*df), ax=axes[1])
            imshow(dx.reshape(nf*df, nf*df), ax=axes[2])
            imshow(dy.reshape(nf*df, nf*df), ax=axes[3])

            # axes[1].quiver(flows[it, :, :, 0], flows[it, :, :, 1])
            # flow = flows[slice(max(it - 50, 0), it)].mean(axis=0)
            flow = flows[it]
            # print flow[:, :, 0]
            # print flow[:, :, 1]
            ax = debug_fig.add_subplot(gs[1])
            ax.quiver(flow[:, :, 0], flow[:, :, 1], units='xy')

            # imshow(dx.reshape(nf*df, nf*df), ax=axes[2])
            # imshow(dy.reshape(nf*df, nf*df), ax=axes[3])
            # (ax.invert_yaxis() for ax in [axes[0], axes[2], axes[3]])

            plt.draw()
            raw_input("Press any key...")


def stereo(events0, events1, t0=0.0, t1=None, debug=False):
    if t1 is None:
        t1 = max(events0['t'].max(), events1['t'].max())

    dt = np.unique(np.diff(np.unique(events0['t']))).min()
    nt = int((t1 - t0) / dt) + 1

    # df = 5
    # nf = int(np.ceil(128. / df))

    images0 = np.zeros((nt, 128, 128))
    images1 = np.zeros((nt, 128, 128))
    # disps = np.zeros((nt, 128, 128))
    disps = np.nan * np.ones((nt, 128, 128))

    for it in range(nt):
        ti = t0 + it * dt
        events0i = events0[close(events0['t'], ti)]
        events1i = events1[close(events1['t'], ti)]

        add_to_image(images0[it], events0i)
        add_to_image(images1[it], events1i)

        matched = np.zeros(len(events1i), dtype=bool)
        for x0, y0, s0, _ in events0i:
            m = ~matched

            # epipolar line
            m &= abs(events1i['y'] - y0) <= 1

            # sign
            m &= events1i['s'] == s0

            n = m.sum()
            if n == 1:
                # disps[it, y0, x0] = x0 - events1i[m]['x']
                disps[it, y0, x0] = x0 - events1i['x'][m]
                matched[m] = 1
            elif n > 1:
                pass
                # print("multi-match")
            else:  # n == 0
                pass

        print(np.nanmin(disps[it]), np.nanmax(disps[it]))

        r = 1
        if debug and it > r:
            plt.figure(101)
            plt.clf()
            axs = [plt.subplot(2, 2, i+1) for i in range(4)]
            imshow(images0[it-r:it+1].sum(0), ax=axs[0])
            imshow(images1[it-r:it+1].sum(0), ax=axs[1])
            img2 = axs[2].imshow(np.nanmean(disps[it-r:it+1], axis=0))
            axs[2].invert_yaxis()
            plt.colorbar(img2, ax=axs[2])

            plt.draw()
            raw_input("Press any key...")


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

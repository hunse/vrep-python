"""
Record DVS data using the V-REP simulator.

Since simulation times are not fast with the kind of time-step
we want for DVS events, we use this file to simulate and record
the events, so that they can be processed in another file.

Using 'dvs-epuck.ttt'
"""
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import vrep

oneshot = vrep.simx_opmode_oneshot

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
    events = []

    lastLen = -1
    lastChar = ''

    show = True
    if show:
        fig = plt.figure(1)
        fig.clf()
        ax = fig.gca()

        image = np.zeros((128, 128), dtype=float)
        ax_image = ax.imshow(image, vmin=-1, vmax=1, cmap='gray', interpolation=None)
        ax.invert_yaxis()

    while True:
        err, data = vrep.simxGetStringSignal(cid, "currentDVS", oneshot)
        if err or len(data) == 0 or (len(data) == lastLen and data[:4] == lastChar):
            time.sleep(0.001)
            continue

        lastLen = len(data)
        lastChar = data[:4]

        # --- Format data
        data = np.array(bytearray(data), dtype=np.uint8)
        data.shape = (-1, 4)

        signs = data[:, 0] >= 128
        data[:, 0] -= signs * 128

        times = 256 * data[:, 3] + data[:, 2]

        # --- Show data
        if show:
            image *= 0.55  # decay image
            for x, y, s, t in zip(data[:, 0], data[:, 1], signs, times):
                image[y, x] += 1 if s else -1

            ax_image.set_data(image)
            plt.draw()

        # --- Store data
        events.append((data[:, :2], signs, times))

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
n = sum(len(xy) for xy, s, t in events)
record = np.zeros(
    n, dtype=[('x', 'u1'), ('y', 'u1'), ('s', 'b'), ('t', 'u2')])

i = 0
for xy, s, t in events:
    n = len(xy)
    record[i:i + n]['x'] = xy[:, 0]
    record[i:i + n]['y'] = xy[:, 1]
    record[i:i + n]['s'] = s
    record[i:i + n]['t'] = t
    i += n

filename = 'dvs.npy'
np.save(filename, record)
print("Saved '%s'" % filename)

"""
Show recorded DVS data (currently from the V-REP simulator)
"""
import numpy as np
import matplotlib.pyplot as plt

import dvs


# filename = 'dvs.npz'
# filename = 'dvs-ball-1ms.npz'
filename = 'dvs-epuck-stereo-2.npz'
events = dvs.load(filename, dt_round=True)

stereo = isinstance(events, tuple) and len(events) == 2
if stereo:
    events0, events1 = events
else:
    events0 = events

plt.figure(101)
diffs = np.diff(np.unique(events0['t']))
plt.hist(diffs)

plt.figure(1)
times = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for i in range(6):
    plt.subplot(2, 3, i+1)
    dvs.imshow(dvs.as_image(events0[dvs.close(events0['t'], times[i])]))
    plt.title("t = %0.3f" % times[i])

plt.figure(2)
axs = [plt.subplot(1, 2, i+1) for i in range(2)] if stereo else plt.gca()
ani = dvs.show(events, stereo=stereo, axs=axs)

plt.show()

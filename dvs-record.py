"""
Record DVS data using the V-REP simulator.

Since simulation times are not fast with the kind of time-step
we want for DVS events, we use this file to simulate and record
the events, so that they can be processed in another file.

Using 'dvs-epuck.ttt'
"""
import numpy as np
import matplotlib.pyplot as plt

import dvs


stereo = False

filename = 'dvs.npz'
# filename = 'dvs-epuck-stereo.npz'
# filename = 'dvs-ball-1ms.npz'

dvs.record(filename, stereo=stereo)

# re-load events, since this scales the time
events = dvs.load(filename)

# plt.figure(1)
# axs = [plt.subplot(1, 2, i+1) for i in range(2)] if stereo else plt.gca()
# ani = dvs.show(events, axs=axs, stereo=stereo)
# plt.show()

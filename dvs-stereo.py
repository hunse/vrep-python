import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import dvs


filename = 'dvs-epuck-stereo-2.npz'
events0, events1 = dvs.load(filename, dt_round=True)

dvs.stereo(events0, events1, debug=True)

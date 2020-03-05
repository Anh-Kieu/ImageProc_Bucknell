'''
Joshua Stough
DIP

Showing sampling/resolution issues.
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale

I = plt.imread('grandCanyon.jpg')


# Scale the image down

sI = rescale(I, .25,
             order=1,
             anti_aliasing=True,
             multichannel=True)

# rescale

reI = rescale(sI, 4,
              order=0,
              anti_aliasing=False,
              multichannel=True)

f, ax = plt.subplots(1,2, sharex=True, sharey=True,
                     figsize=(12, 3))

ax[0].imshow(I, interpolation=None)
ax[0].set_title('Original')

ax[1].imshow(reI, interpolation=None)
ax[1].set_title('Quarter Res')

# [a.axes.get_xaxis().set_visible(False) for a in ax]
# [a.axes.get_yaxis().set_visible(False) for a in ax]
plt.tight_layout()

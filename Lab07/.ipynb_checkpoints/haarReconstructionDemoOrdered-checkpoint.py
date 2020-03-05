"""
Joshua Stough
DIP

Decompose and partially reconstruct according to the Haar
basis.
"""

from waveletUtil import *

print('haarReconstructionDemo: Works only on 64 pixel images...')

I = plt.imread('surprise.png')
GI = 0.2989 * I[..., 0] + 0.5870 * I[..., 1] + 0.1140 * I[..., 2]

#H is the 8x8 Haar matrix
H = makeHaarMatrix(8)

#The transform image, an image of coefficients wrt the Haar basis.
T = np.matmul(H, np.matmul(GI, H.transpose()))

#RI will represent the reconstructed image as we add back more
#Haar patterns
RI = np.zeros(GI.shape) #Still should be 8x8


#Visual
fh, axh = plt.subplots(8,8, figsize=(10,10))
fr, axr = plt.subplots(8,8, figsize=(10,10))


# We're going to reconstruct according to distance from the
# 0,0 (the first Haar basis, average calculator). Notice,
# this order is independent of the actual image data.
xs = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
coords = np.concatenate([np.expand_dims(c, axis=1) for c in
                         [x.ravel() for x in xs]], axis=1)
dists = np.sum(coords*coords, axis=1)
darg = np.argsort(dists) #sorts in increasing order

#If we were to use magnitude of the coefficient...need to ravel in
#column-major, like we set up coords.
#mags = T.ravel(order='F')
#darg = np.argsort(mags)
#darg.reverse()

#c = 1 #used to display the order of reconstruction.

for ind in darg:
    i,j = coords[ind] #coords[darg[x]] if x in range(len(darg))

    #Construct that Haar basis and display it
    Bij = np.outer(H[i, :], H[j, :])
    axh[i][j].imshow(Bij, cmap='gray', vmin=-1, vmax=1)
    axh[i][j].axes.get_xaxis().set_visible(False)
    axh[i][j].axes.get_yaxis().set_visible(False)
    # axh[i][j].set_title('c_%d_%d: %6.3f' % (i, j, T[i,j]))
    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.text
    axh[i][j].text(0, 6, r'$\pi:%6.3f$' % T[i, j], fontsize=6, color='cyan')

    # Add the amount of that basis that was in the original image to
    # the running total, or reconstruction.
    RI = RI + T[i, j] * Bij
    axr[i][j].imshow(RI, cmap='gray', vmin=0, vmax=1)
    axr[i][j].axes.get_xaxis().set_visible(False)
    axr[i][j].axes.get_yaxis().set_visible(False)
    #axr[i][j].text(0, 6, '%d' % c, fontsize=8, color='cyan')
    #The order of reconstruction isn't that informative, but useful for debugging.
    #c += 1


plt.show()

fh.canvas.set_window_title('Haar Basis Images and Coefficients')
fh.tight_layout()

fr.canvas.set_window_title('Partial Reconstructions from Large Scale to Small')
fr.tight_layout()

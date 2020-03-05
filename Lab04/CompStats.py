import huffTreeUtilities as hf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def encode(I):
    Ic = I.copy()
    Ic_red = hf.loadHuffableImage(Ic[..., 0])
    red_chan = Ic[...,0]
    encoder_red, decoder_red = hf.buildHuffPair(red_chan)
    en_red = ''.join(encoder_red[pix] for pix in red_chan.ravel())

    Ic_green = hf.loadHuffableImage(Ic[..., 1])
    green_chan = Ic[...,1]
    encoder_green, decoder_green = hf.buildHuffPair(green_chan)
    en_green = ''.join(encoder_green[pix] for pix in green_chan.ravel())

    Ic_blue = hf.loadHuffableImage(Ic[..., 2])
    blue_chan = Ic[...,2]
    encoder_blue, decoder_blue = hf.buildHuffPair(blue_chan)
    en_blue = ''.join(encoder_blue[pix] for pix in blue_chan.ravel())
    return en_red, en_green, en_blue

def printStatsChannel(encoded, origin, name):
    '''
    Prints the info of the encoding for each channel
    input: encoded: the encoded string
    origin: the raveled color channel
    name: name of the channel
    '''
    print("Channel " + name + " statistics:")
    print("Load Hufffable Image: Setting range to [0,255]")
    # total entropy = - sum(bin 0 -> bin 255) of probability(event) * log2(probability(event))
    # then just compute the size
    # then the encoded file, comparing the bits/pixel :D, should be roughly the same with the entropy
    freq, bins = np.histogram(origin, bins = np.arange(257))
    print(name + " channel entropy is " + str(entropy(freq, base=2)))
    print("Size at 8-bit encoding: " + str(len(origin)/1000) + " KB")
    print("Size with huff encoding: " + str(len(encoded)/8000) + " KB or " + str(len(encoded)/len(origin)) + " bits per pixel.")

def getCompressionStats(image):
    '''
    Print out the statistics for the compression
    '''
    I = plt.imread(image, 'uint8')
    r_comp, g_comp, b_comp = encode(I)
    printStatsChannel(r_comp, I[...,0].ravel(), "Red")
    printStatsChannel(g_comp, I[...,1].ravel(), "Green")
    printStatsChannel(b_comp, I[...,2].ravel(), "Blue")

# getCompressionStats('happyFace.png')

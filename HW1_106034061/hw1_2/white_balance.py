
import numpy as np
import cv2
def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    #h, w, c=img.shape
   # mask=np.zeros(((h, w, c)))
   # print(mask)
    
    pattern = pattern.upper()

    channels = dict((channel, np.zeros(img.shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        if(channel=='R'):
            channels[channel][y::2, x::2] = fr

        elif(channel=='B'):
            channels[channel][y::2, x::2] = fb
        else:
            channels[channel][y::2, x::2] = 1
    #print(channels)
    output=channels['R']+channels['G']+channels['B']
    #print(output)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return output
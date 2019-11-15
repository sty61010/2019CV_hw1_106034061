
import cv2
import skimage
import numpy as np

def BGR2RGB(img):
    '''
    Input:
        img: H*W*3, input BGR image
    Output:
        output: H*W*3, output RGB image
    '''
    b, g, r = cv2.split(img)
    output = cv2.merge([r, g, b])
    return output

def RGB2BGR(img):
    '''
    Input:
        img: H*W*3, input RGB image
    Output:
        output: H*W*3, output BGR image
    '''
    r, g, b = cv2.split(img)
    output = cv2.merge([b, g, r])
    return output

def RGB2XYZ(img):
    '''
    Input:
        img: H*W*3, input RGB image
    Output:
        output: H*W*3, output CIE XYZ image
    '''
    output = skimage.color.rgb2xyz(img)
    return output

def XYZ2RGB(img):
    '''
    Input:
        img: H*W*3, input CIE XYZ image
    Output:
        output: H*W*3, output RGB image
    '''
    output = skimage.color.xyz2rgb(img)
    return output


def color_correction(img, ccm):
    '''
    Input:
        img: H*W*3 numpy array, input image
        ccm: 3*3 numpy array, color correction matrix 
    Output:
        output: H*W*3 numpy array, output image after color correction
    '''
    ########################################################################
    # TODO:                                                                #
    #   Following the p.22 of hw1_tutorial.pdf to get P as output.         #
    #                                                                      #
    ########################################################################
    h,w,c=img.shape
    r, g, b = cv2.split(img)

    r=np.reshape(r,(h*w,1))
    g=np.reshape(g,(h*w,1))
    b=np.reshape(b,(h*w,1))
    #print(r)

    #print(img)
    output=np.concatenate((r,g,b),axis=1)
    output=np.dot(output,ccm)
    #print(output)
    r=np.reshape(output[:,0],(h,w))
   # print(r)
    g=np.reshape(output[:,1],(h,w))
    b=np.reshape(output[:,2],(h,w))
    output=cv2.merge([r,g,b])
    #print(output)
    #cv2.imshow("color_correction",output)
    
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    
    #### Prevent the value larger than 1 or less than 0
    output = np.clip(output, 0, 1)
    return output

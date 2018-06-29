"""
Created on: 2018-Jun-28
File: dominant_color.py
'''
@author: Alvin(Xinyao) Sun
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pylab as plt


def get_dominant_colors(img, rects):
    """Return dominat colors from a list of rectangles according to a given image
    
    Args:
        img (numpy array): uint8 numpy array with RGB color format
        rets (numpy array): a list of rectangles with format of [ [x, y, width, height], [...], [...] ]
    
    Returns:
        list: a list of colors (R, G, B) following the same order of rectangles
    """

    d_colors = []
    for rect in rects:
        croped = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        
        # Quantizaion of colors to 20 levels
        ks = (KMeans(n_clusters=20).fit(np.reshape(croped,[-1,3])))

        # Get most frequent color 
        (values,counts) = np.unique(ks.labels_,return_counts=True)
        ind=np.argmax(counts)

        # Round and clip most frequent keam center
        c_center = np.clip(np.round(ks.cluster_centers_[values[ind]]),0,255).astype("uint8") 

        # Treat it as dominant color
        d_colors.append(c_center)

         
    return d_colors
    
#################################
#                               #
#         Test Script           #
#                               #
#################################
if __name__ == "__main__":
    img = cv2.imread("bottle.jpg")
    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
    r = cv2.selectROI(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    colors = get_dominant_colors(img,[r])
    z = np.ones((40,40,3),dtype="uint8")*colors[0]
    plt.imshow(z),plt.show()



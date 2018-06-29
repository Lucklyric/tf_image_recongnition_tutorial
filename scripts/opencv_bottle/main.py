import cv2
import numpy as np
import matplotlib.pylab as plt

# src = cv2.imread("bottle.jpg")

# image = cv2.imread('bottle.jpg',cv2.IMREAD_COLOR)
# image = cv2.bilateralFilter(image,9,75,75)
# original = np.copy(image)
# if image is None:
#     print ('Can not read/find the image.')
#     exit(-1)

# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# H,S,V = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
# V = V * 2

# hsv_image = cv2.merge([H,S,V])
# image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# image = cv2.medianBlur(image,5)
# plt.figure(), plt.imshow(image)
# plt.show()

# Dx = cv2.Sobel(image,cv2.CV_8UC1,1,0)
# Dy = cv2.Sobel(image,cv2.CV_8UC1,0,1)
# M = cv2.addWeighted(Dx, 1, Dy,1,0)

# plt.subplot(1,3,1), plt.imshow(Dx, 'gray'), plt.title('Dx')
# plt.subplot(1,3,2), plt.imshow(Dy, 'gray'), plt.title('Dy')
# plt.subplot(1,3,3), plt.imshow(M, 'gray'), plt.title('Magnitude')
# plt.show()

# ret, binary = cv2.threshold(M,10,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# plt.figure(), plt.imshow(binary, 'gray')
# plt.show()

# binary = binary.astype(np.uint8)
# binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
# edges = cv2.Canny(binary, 50, 100)
# plt.figure(), plt.imshow(edges, 'gray')
# plt.show()

# lines = cv2.HoughLinesP(edges,1,3.14/180,50,20,10)[0]
# plt.show()
# output = np.zeros_like(M, dtype=np.uint8)
# for line in lines:
#     cv2.line(output,(line[0],line[1]), (line[2], line[3]), (100,200,50), thickness=2)
# plt.figure(), plt.imshow(output, 'gray')

# points = np.array([np.transpose(np.where(output != 0))], dtype=np.float32)
# rect = cv2.boundingRect(points)
# cv2.rectangle(original,(rect[1],rect[0]), (rect[1]+rect[3], rect[0]+rect[2]),(255,255,255),thickness=2)
# original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
# plt.figure(), plt.imshow(original,'gray')
from skimage.transform import rescale 

image = cv2.imread('bottle.jpg',0)
image = cv2.resize(image,(512,512))

# image = cv2.bilateralFilter(image,5,35,35)


# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# H,S,V = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
# V = V * 2

cimg = cv2.imread('bottle.jpg',1)
cimg = cv2.resize(cimg,(512,512))
# hsv_image = cv2.merge([H,S,V])
# image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
img = image
# ve
img[cimg[:,:,0]<180]= 0

img = cv2.medianBlur(image,11)
# img = cv2.imread('bottle.jpg',0)

# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
plt.imshow(img,cmap="gray",vmin=0,vmax=255),plt.show()
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,3,35,
#                             param1=20,param2=100,minRadius=5,maxRadius=55)

# circles = np.uint16(np.around(circles))

# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# print(len(circles[0,:]))
# cv2.imshow('detected circles',cimg)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# th, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img[img>0] = 255 
bw = img
plt.imshow(img,cmap="gray"),plt.show()

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
# dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
# borderSize = 75
# distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
#                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
# gap = 10                                
# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
# kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
#                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
# distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
# nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
# mn, mx, _, _ = cv2.minMaxLoc(nxcor)
# th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(cimg, contours, -1, (0,255,0), 3)
# hull = cv2.convexHull(contours[3])

# for each contour
for cnt in contours:
    # get convex hull
    hull = cv2.convexHull(cnt)
    # draw it in red color
    cv2.drawContours(cimg, [hull], -1, (0, 0, 255), 3)

# peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
# for i in range(len(contours)):
#     x, y, w, h = cv2.boundingRect(contours[i])
#     _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
#     cv2.circle(cimg, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
#     cv2.rectangle(cimg, (x, y), (x+w, y+h), (0, 255, 255), 2)
#     cv2.drawContours(cimg, contours, i, (0, 0, 255), 2)

cv2.imshow('circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
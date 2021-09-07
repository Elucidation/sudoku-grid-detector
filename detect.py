import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
# img_orig = cv2.imread('full1.jpg')
t_start = time.time()
img_orig = cv2.imread('crop1.jpg')

# Resize images to max height of 1200
delta = 1024.0 / img_orig.shape[0] 
print(f'Resizing {img_orig.shape} x {delta}')
img = cv2.resize(img_orig,None,fx=delta, fy=delta , interpolation = cv2.INTER_CUBIC)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray = cv2.GaussianBlur(img_gray,(5,5),0)

# Thresholded + Inverted so black lines are now white.
thresh1 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,21,15)

# Find enclosing rectangle
contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

simple_contours = []
for contour in contours:
  # If it covers at least 1/6th of the image area
  if cv2.contourArea(contour) > thresh1.shape[0]*thresh1.shape[1]/6:
    epsilon = 0.1*cv2.arcLength(contour,True)
    contour = cv2.approxPolyDP(contour,epsilon,True)
    # contour = cv2.convexHull(contour) # Orders points nicely
    if contour.shape[0] != 4:
      continue
    simple_contours.append(contour)

if len(simple_contours) != 1:
  print(simple_contours)
  raise Exception(f'Didn\'t find exactly one contour: {len(simple_contours)}')

# overlay = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
overlay = img.copy()
cv2.drawContours(overlay, simple_contours, -1, (00, 255, 0), 10)

def order_points(pts):
  out = np.zeros((4, 2), dtype=np.float32)
  s = pts.sum(axis = 1)
  out[0] = pts[np.argmin(s)]
  out[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis = 1)
  out[1] = pts[np.argmin(diff)]
  out[3] = pts[np.argmax(diff)]
  return out

# Generate warped image
input_pts = np.float32(simple_contours[0][:,0,:])
input_pts = order_points(input_pts)
tilesize = 64 # px
output_pts= np.float32([
  [0,0],
  [1,0],
  [1,1],
  [0,1]]) * tilesize*9
M = cv2.getPerspectiveTransform(input_pts/delta,output_pts)
# warped = cv2.warpPerspective(thresh1,M,(tilesize*9, tilesize*9),flags=cv2.INTER_LINEAR)
warped = cv2.warpPerspective(img_orig,M,(tilesize*9, tilesize*9),flags=cv2.INTER_LINEAR)

warped = cv2.adaptiveThreshold(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,21,50)

# warped img size is width = height = tilesize * 9
# remove lines arbitrarily
buffer_px = int(tilesize/6)
for i in range(10):
  warped[:,max(0,i*tilesize-buffer_px):min(tilesize*9, i*tilesize+buffer_px)] = 0
  warped[max(0,i*tilesize-buffer_px):min(tilesize*9, i*tilesize+buffer_px),:] = 0

tot_time = time.time()-t_start
print(f'Took {tot_time:.3f} seconds to load image, process and draw')
plt.figure(figsize=(20,10))
plt.subplot(1,4,1),plt.imshow(img,vmin=0,vmax=255)
plt.title('Original Image')
plt.xticks([]),plt.yticks([])

plt.subplot(1,4,2),plt.imshow(thresh1, 'gray',vmin=0,vmax=255)
plt.plot(simple_contours[0][:,0,0], simple_contours[0][:,0,1],'go', markersize=15)
plt.title('ADAPTIVE_THRESH_GAUSSIAN_C')
plt.xticks([]),plt.yticks([])

plt.subplot(1,4,3),plt.imshow(overlay,vmin=0,vmax=255)
plt.title('OVERLAY')
plt.xticks([]),plt.yticks([])

plt.subplot(1,4,4),plt.imshow(warped,'gray',vmin=0,vmax=255)
plt.title('WARPED')
plt.xticks([]),plt.yticks([])

plt.tight_layout()
plt.show()
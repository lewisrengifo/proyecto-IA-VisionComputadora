import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('cafe.jpg', 0)
img2 = cv2.imread('Starbucks-logo.png',0)
#resize
scale_percent = 20 # percent of original size
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
img2Resized = cv2.resize(img2,dim, interpolation= cv2.INTER_AREA)
print(f'shape: {img2Resized.shape}')
cv2.imshow('cafe', img1)
cv2.imshow('logo', img2Resized)
cv2.waitKey(0)
#Orb creator
orb = cv2.ORB_create()

#find keypoints
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2Resized, None)

# create BFmATHCHER
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

#MatchDescriptors
matches = bf.match(des1, des2)

#Sort the ordes
matches = sorted(matches, key = lambda x:x.distance)

#Draw the first 30 matches
img3 = cv2.drawMatches(img1, kp1, img2Resized, kp2, matches[:30], None, flags=2)
plt.imshow(img3)
plt.show()
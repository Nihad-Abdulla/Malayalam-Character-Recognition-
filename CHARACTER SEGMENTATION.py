#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
canny = cv2.Canny(skeleton, 120, 255, 1)
dilate = cv2.dilate(canny, None, iterations=1)

cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] 
sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[1] * image.shape[1])

orig = skeleton.copy()
plt.imshow(orig,'gray')
i = 0
for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv2.contourArea(cnt) < 500):
        continue

    # Filtered countours are detected
    x,y,w,h = cv2.boundingRect(cnt)

    # Taking ROI of the cotour
    roi = skeleton[y:y+h, x:x+w]
    
    # Resize ROI to 128x128
    roi = cv2.resize(roi, (128, 128))

    # Mark them on the image if you want
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),3)

    # Save your contours or characters
    cv2.imwrite("C:/input/character segmentation/e" + str(i) + ".png", roi)
    
    i = i + 1
    
cv2.imwrite("C:\input\malayalam\e1 character.jpg",orig)
plt.figure(figsize=(15,15))
plt.title("Character Segmetation")
plt.imshow(orig,'gray')
cnts.clear()


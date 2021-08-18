import sys
import os
import time

import insightface
import urllib
import urllib.request
import cv2
import numpy as np

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def draw_boxes(img, boxes, points):
    im_copy = img.copy()
    i = 0
    for b in boxes:
        cv2.rectangle(im_copy, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        cv2.circle(im_copy, (points[i][0][0], points[i][0][1]), 3, (0, 0, 255), 1)
        cv2.circle(im_copy, (points[i][1][0], points[i][1][1]), 3, (0, 0, 255), 1)
        cv2.circle(im_copy, (points[i][2][0], points[i][2][1]), 3, (0, 0, 255), 1)
        cv2.circle(im_copy, (points[i][3][0], points[i][3][1]), 3, (0, 0, 255), 1)
        cv2.circle(im_copy, (points[i][4][0], points[i][4][1]), 3, (0, 0, 255), 1)
        cv2.imshow("./tmp.jpeg", im_copy)
        cv2.waitKey(0)

def save_faces(img, boxes, points, img_name, path):
    im_copy = img.copy()
    i = 0
    for b in boxes:
        i=i+1
        x1 = int(b[0])
        x2 = int(b[2])
        y1 = int(b[1])
        y2 = int(b[3])

        size = im_copy.shape
        if x1 < 0:
            x1 = 0
        if x2 > size[1]:
            x2 = size[1]
        if y1 < 0:
            y1 = 0
        if y2 > size[0]:
            y2 = size[0]
        
        im_save = im_copy[y1:y2, x1:x2]
        img_name_split = img_name.split(".")
        print (path, img_name_split[0], str(i))
        im_save_name = path+img_name_split[0]+"_"+str(i)+".jpg"
        cv2.imwrite(im_save_name, im_save)

def main():
    args = sys.argv[1:]
    if len(args)<2:
        print ("argument(s) missed!")

    in_img  = args[0]
    path = args[1]

    img = cv2.imread(in_img)
    model = insightface.model_zoo.get_model('retinaface_r50_v1')

    model.prepare(ctx_id = -1, nms=0.4)

    t1 = time.time()
    bbox, landmark = model.detect(img, threshold=0.5, scale=1.0)
    t2 = time.time()

    print ("  ")
    print ("It took ", round(t2-t1, 4), "s to detect faces in the input image!")

    # draw_boxes(img, bbox, landmark)
    save_faces(img, bbox, landmark, in_img, path)

    print ("  ")
    print ("Detection Finished!", in_img)

if __name__ == '__main__':
    main()

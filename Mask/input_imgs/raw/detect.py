import insightface
import urllib
import urllib.request
import cv2
import os
import sys
import time
import PIL.Image as imgs
import numpy as np

def draw_boxes(boxes, plotname, outname):
    i = 0
    for b in boxes: 
        raw_plot = imgs.open(plotname)
        plot_range = (boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])
        cut_plot = raw_plot.crop(plot_range)
        cut_plot.save(outname+"_%s.jpg"%i)
        i = i + 1
        cv2.waitKey(0)

def main():
    plotname = sys.argv[1]
    outname = sys.argv[2]
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = -1, nms=0.4)
    t1 = time.time()
    img = cv2.imread(plotname)
    boxes, landmark = model.detect(img,threshold=0.3, scale=1.0)
    t2 = time.time()
    print ("t2 - t1 ===>>>s", t2 - t1)
    draw_boxes(boxes, plotname, outname)
    print ("saved"+plotname)

if __name__=='__main__':
    main()

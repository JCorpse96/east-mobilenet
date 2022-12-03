import os
import cv2
import torch
import argparse
import numpy as np
import tensorflow as tf

import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)

from eval import load_model
from utils.util import resize_image
from utils.util import detect
from utils.util import sort_poly
from utils.util import min_max_points
from utils.util import binary_sort
from utils.util import get_targets
from utils.util import min_max_points_cnt
from utils.util import remove_inbound_boxes

def demo(model, img_path, save_path, with_gpu):
    alphaNet = tf.saved_model.load("alphaNet/1/")
    infer = alphaNet.signatures["serving_default"]
    target_classes = get_targets()
    show_words = True
    show_image = True
    with torch.no_grad():
        im = cv2.imread(img_path)
        img = im[:, :, ::-1].copy()
        im_resized, (ratio_h, ratio_w) = resize_image(img, 512)
        im_resized = im_resized.astype(np.float32)
        im_resized = torch.from_numpy(im_resized)
        if with_gpu:
            im_resized = im_resized.cuda()

        im_resized = im_resized.unsqueeze(0)
        im_resized = im_resized.permute(0, 3, 1, 2)

        score, geometry = model.forward(im_resized)

        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.detach().cpu().numpy()
        geometry = geometry.detach().cpu().numpy()

        boxes = detect(score_map=score, geo_map=geometry)
        
        if len(boxes) > 0:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
    
        if boxes is not None:
            for box in boxes:
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=2)

        #print(boxes)
        
        words = ""
        
        boxes = [min_max_points(box) for box in boxes]
        boxes = binary_sort(boxes)[0]

        
        
        #print()
        #print(boxes)

        for box in boxes:
            
            letters = im[box[0][1]:box[1][1], box[0][0]:box[1][0], :].copy()
            grey = cv2.cvtColor(letters,cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(grey,(5,5),0)
            #blur = 255 - cv2.bitwise_not(grey)
            imgBin = cv2.Canny(blur,150, 220, 3)
            #(t,imgBin) = cv2.threshold(canny,0,255,cv2.THRESH_OTSU)
            #imgBin = cv2.bitwise_not(imgBin)
            strelDilateDetect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)) #(cv2.MORPH_ELLIPSE, (4,4))
            strelOpenDetect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) #(cv2.MORPH_ELLIPSE, (11,11))
            imgBin = cv2.dilate(imgBin,strelDilateDetect, iterations=3)
            #imgBin = cv2.morphologyEx(imgBin,cv2.MORPH_OPEN,strelOpenDetect)
            #imgBin = 255 - imgBin
            
           
            (cnt, heirarchy) = cv2.findContours(imgBin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #print(cnt[0])
            print(heirarchy)
            cntParent = [cnt[i] for i in range(len(cnt)) if heirarchy[0][i][3] == -1]
            cntChild = [cnt[i] for i in range(len(cnt)) if heirarchy[0][i][3] != -1]
            #cv2.drawContours(imgBin, cntParent, -1, color=(255, 255, 255), thickness=cv2.FILLED)
            #cv2.drawContours(imgBin, cntChild, -1, color=(0, 0, 0), thickness=cv2.FILLED)
            cntParent = [min_max_points_cnt(cont) for cont in cntParent]
            cntParent = binary_sort(cntParent)[0]
            inbounds = remove_inbound_boxes(cntParent)
            cntParent = [cntParent[i] for i in range(len(cntParent)) if inbounds[i] != 1]
            print("n of contours",len(cntParent))
            print("inbounds",inbounds)

            (t, post_proc) = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
            strelDilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) #(cv2.MORPH_ELLIPSE, (4,4))
            strelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) #(cv2.MORPH_ELLIPSE, (11,11))
            post_proc = cv2.dilate(post_proc, strelDilate)
            post_proc = cv2.morphologyEx(post_proc,cv2.MORPH_OPEN,strelOpen)

            word = ""
            i = 0
            for cont in cntParent:
                print("heirarchy", heirarchy[0][i])
                ((x1,y1) , (x2,y2)) = cont
                letter = 255 - cv2.bitwise_not(post_proc)[y1:y2, x1:x2].copy()
                #letter = imgBin[y1:y2, x1:x2].copy()
                #print(((x1,y1) , (x2,y2)))
                letters = cv2.rectangle(letters, (x1,y1),(x2,y2), (0,255,0),2)

                letter = cv2.resize(letter,(20,20),interpolation=cv2.INTER_AREA).astype(np.float32)
                img_pred = np.expand_dims(letter, axis=(0,3))
                outputs = infer(tf.constant(img_pred))
                y_test_pred = tf.argmax(outputs['dense_1'].numpy(), axis=1)
                index = y_test_pred.numpy()[0]
                pred = target_classes[index][str(index)]
                print(pred)
                word += pred

                i += 1

                cv2.imshow("letter", letter)
                cv2.waitKey(0)
                cv2.destroyWindow("letter")

            #cv2.drawContours(letters, cnt, -1, (0,255,0), 2)

            """
            (numLabels, labels, detection, centroids) = cv2.connectedComponentsWithStats(imgBin)

            detection = detection[1:, 0:4]
            detection = binary_sort(np.reshape(detection, (len(detection),2,2)))[0]
            
            
            
            for i in range(len(detection)):
                (x, y, w, h) = (detection[i][0][0], detection[i][0][1], detection[i][1][0], detection[i][1][1])
                letter = imgBin[y:y+h, x:x+w].copy()
                #letters = cv2.rectangle(letters, (x,y),(x+w,y+h), (0,255,0),2)
                letter = cv2.resize(letter,(20,20),interpolation=cv2.INTER_AREA).astype(np.float32)
                
                img_pred = np.expand_dims(letter, axis=(0,3))
                outputs = infer(tf.constant(img_pred))
                y_test_pred = tf.argmax(outputs['dense_1'].numpy(), axis=1)
                index = y_test_pred.numpy()[0]
                pred = target_classes[index][str(index)]
                print(pred)
                word += pred


                cv2.imshow("letter", letter)
                cv2.waitKey(0)
                cv2.destroyWindow("letter")
            """

            cv2.putText(img, word, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            print("\nWORD:",word)
            words += word + " "
            
            if show_words:
                cv2.imshow("blured", blur)
                cv2.imshow("grey", cv2.bitwise_not(post_proc))
                cv2.imshow("cropped", letters)
                cv2.imshow(word, imgBin)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("\nPREDICTION:", words)
        if show_image:
            cv2.imshow("predicted",resize_image(img[:, :, ::-1],928)[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        cv2.imwrite(os.path.join(save_path, 'result-{}'.format(img_path.split('/')[-1])), img[:, :, ::-1])


def main(args: argparse.Namespace):
    with_gpu = True if torch.cuda.is_available() else False
    save_path = "demo"
    model_path = args.model
    img_path = args.image
    model = load_model(model_path, with_gpu)
    demo(model, img_path, save_path, with_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('-m', '--model', default=None, type=str, required=True, help='path to model')
    parser.add_argument('-i', '--image', default=None, type=str, required=True, help='input image')
    args = parser.parse_args()
    main(args)

import os
import cv2
import torch
import numpy as np
import json
import math

from utils.nms import nms_locality
from english_words import english_words_lower_alpha_set


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def polygon_area(poly):
    """
    compute area of a polygon
    :param poly:
    :return:
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    """1e-5
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle_rbox(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    boxes = nms_locality(boxes.astype(np.float32), nms_thres)
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    if boxes.shape[0] == 0:
        return np.array([])

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    # boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes


def predict(im_fn, model, with_img=False, output_dir=None, with_gpu=False):
    im = cv2.imread(im_fn.as_posix())[:, :, ::-1]
    im_resized, (ratio_h, ratio_w) = resize_image(im)
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
        # scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    if boxes is not None:
        res_file = os.path.join(output_dir, 'res_{}.txt'.format(
            os.path.basename(im_fn.as_posix()).split('.')[0]))

        with open(res_file, 'w') as f:
            for box in boxes:
                # to avoid submitting errors
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                ))
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=1)

    if with_img:
        img_path = os.path.join(output_dir, 'img', im_fn.name)
        cv2.imwrite(img_path, im[:, :, ::-1])


#ADDED CUSTOM METHODS
def min_max_points(box, y_max=10000, x_max=10000):
    np_box = np.array(box)

    avg_x_min = max(0,int((np_box[0][0] + np_box[3][0]) / 2))
    avg_y_min = max(0,int((np_box[0][1] + np_box[1][1]) / 2))

    avg_x_max = min(x_max,int((np_box[1][0] + np_box[2][0]) / 2))
    avg_y_max = min(y_max,int((np_box[2][1] + np_box[3][1]) / 2))

    x_min_point = max(0, int(np.amin(np_box[:, 0])))
    y_min_point = max(0, int(np.amin(np_box[:, 1])))

    x_max_point = min(x_max, int(np.amax(np_box[:, 0])))
    y_max_point = min(y_max, int(np.amax(np_box[:, 1])))

    return np.array([[x_min_point, y_min_point], [x_max_point, y_max_point]])

def init_leaf():
    leaf = {
        "value" : None,
        "left" : None,
        "right" : None
    }
    return leaf

def add_box_leaf(tree, value, word=0, letter=1):
    if tree["value"] is None:
        tree["value"] = value

    elif value[0][word] < tree["value"][0][word]:
        if tree["left"] is None:
            tree["left"] = init_leaf()

        add_box_leaf(tree["left"], value, word, letter)

    elif value[0][word] > tree["value"][0][word]:
        if tree["right"] is None:
            tree["right"] = init_leaf()

        add_box_leaf(tree["right"], value, word, letter)

    else:
        if value[0][letter] < tree["value"][0][letter]:
            if tree["left"] is None:
                tree["left"] = init_leaf()

            add_box_leaf(tree["left"], value, word, letter)

        elif value[0][letter] >= tree["value"][0][letter]:
            if tree["right"] is None:
                tree["right"] = init_leaf()

            add_box_leaf(tree["right"], value, word, letter)
        

def reshape_tree(tree):
    if tree["left"] is None:
        tree["value"] = np.array(tree["value"]).flatten().tolist()
        if tree["right"] is not None:
            reshape_tree(tree["right"])

    else:
        reshape_tree(tree["left"])
        tree["value"] = np.array(tree["value"]).flatten().tolist()
        if tree["right"] is not None:
            reshape_tree(tree["right"])

def flatten_tree(tree, list):
    if tree["left"] is None:
        list.append(tree["value"])
        if tree["right"] is not None:
            flatten_tree(tree["right"], list)

    else:
        flatten_tree(tree["left"], list)
        list.append(tree["value"])
        if tree["right"] is not None:
            flatten_tree(tree["right"], list)


def binary_sort(list, word=0, letter=1):
    tree_4dim = init_leaf()
    ordered_points = []
    for point in list:
        add_box_leaf(tree_4dim, point, word, letter)

    flatten_tree(tree_4dim, ordered_points)
    
    return np.array(ordered_points), tree_4dim

def get_targets():
    f = open("alphaNet/1/letterMapping.json")
    mappings = json.load(f)
    f.close()
    return mappings['labels']


def boxes_stats(boxes):
    areas = np.array([box[1][0]*box[1][1] for box in boxes])
    average = sum(areas) / len(areas)
    deviations = np.square(areas - average)
    variance = sum(deviations) / len(deviations)
    standard_deviation = math.sqrt(variance)
    
    return areas, average, deviations, standard_deviation

def merge_y_axis_boxes(boxes):
    merged = []
    merge = False
    i = 0
    temp_box = boxes[i]
    for j in range(1, len(boxes)):
        if i < len(boxes) - 1:
            if merge == False:
                temp_box = boxes[i]
        
            else:
                temp_box = merged[-1]

            if x_intersection(temp_box, boxes[j]) < 0.6:
                if merge == False:
                    merged.append(temp_box)
                    if j == len(boxes) - 1:
                        merged.append(boxes[j])
                        break
            
                else:
                    i += 1
                    merge = False
        
            else:
                merged_boxes = merge_boxes(temp_box, boxes[j])
                merged.append(merged_boxes)
                merge = True
                i -= 1

            i += 1
        
        else :
            break

    return merged


def x_intersection(box1, box2):
    intersect = (min(box1[0][0] + box1[1][0], box2[0][0] + box2[1][0]) - max(box1[0][0], box2[0][0])) / max(box1[1][0], box2[1][0])
    
    return intersect

def merge_boxes(box1, box2):
    x_min_point = min(box1[0][0], box2[0][0])
    y_min_point = min(box1[0][1], box2[0][1])

    x_max_point = max(box1[0][0] + box1[1][0], box2[0][0] + box2[1][0])
    y_max_point = max(box1[0][1] + box1[1][1], box2[0][1] + box2[1][1])

    return [[x_min_point, y_min_point], [x_max_point - x_min_point, y_max_point - y_min_point]]

def create_prompt(words):
    return """Return only the words I pass to you corrected and only one suggestion.
            The words to correct are: {}
            """.format(words)

def check_dictionary(word):
    return word in english_words_lower_alpha_set

def dictionary(lim=10):
    print("dictionary size ", len(english_words_lower_alpha_set))
    i = 0
    for word in english_words_lower_alpha_set:
        if i == lim:
            break

        print(word)
        i += 1
from shapely.geometry import Polygon
import pyclipper
import cv2
import numpy as np
import copy

det_mean = [0.485, 0.456, 0.406]
det_std = [0.229, 0.224, 0.225]


def draw_det(image, dt_boxes):
    for box in dt_boxes:
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [box], True, color=(255, 255, 0), thickness=2)
    return image


def preprocess_boxes(dt_boxes, ori_im):
    def get_rotate_crop_image(img, points):
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(dt_boxes):
        num_boxes = dt_boxes.shape[0]
        # print(num_boxes)
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    img_crop_list = []
    dt_boxes = sorted_boxes(dt_boxes)
    # print(dt_boxes)
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
    return dt_boxes, img_crop_list


class DetPreProcess:
    def __init__(self):
        pass

    def NormalizeImage(self, data):
        scale = 1.0 / 255
        mean = det_mean
        std = det_std
        order = 'hwc'
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype('float32')
        std = np.array(std).reshape(shape).astype('float32')
        return (data.astype('float32') * scale - mean) / std

    def get_inputs(self, image, do_normalize=True):
        if isinstance(image, str):
            src_frame = cv2.imread(image)
        else:
            src_frame = image
        src_frame = cv2.resize(src_frame, (960, 960))
        data = src_frame
        if do_normalize:
            data = self.NormalizeImage(data)
            data = data.transpose((2, 0, 1))
        return data, src_frame


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def box_score_fast(bitmap, _box):
    '''
    box_score_fast: use bbox mean score as the mean score
    '''
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


class Det:
    def __init__(self,
                 thresh=0.5,
                 box_thresh=0.5,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 target_size=None):
        if target_size is None:
            target_size = [960, 960]
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])
        self.target_size = target_size

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(outs)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = box_score_fast(pred, points.reshape(-1, 2))
            # print(score)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def predict(self, result, src_image, save_path):
        result = result[0]
        pred = result[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w = self.target_size[0], self.target_size[1]
            if self.dilation_kernel is not None:
                # print("true")
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                # print("flase")
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        boxes = boxes_batch[0]['points']
        dt_boxes = self.filter_tag_det_res(boxes)
        dt_boxes, img_crop_list = preprocess_boxes(dt_boxes, src_image)
        tmp = draw_det(src_image, dt_boxes)
        cv2.imwrite(save_path, tmp)
        return dt_boxes, img_crop_list
    def filter_tag_det_res(self, dt_boxes):
        img_height, img_width = self.target_size[0],self.target_size[1]
        dt_boxes_new = []
        for box in dt_boxes:
            box = order_points_clockwise(box)
            box = clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

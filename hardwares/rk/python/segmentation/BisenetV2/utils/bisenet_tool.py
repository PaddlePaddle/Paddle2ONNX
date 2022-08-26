import numpy as np
import cv2


def preprocess(image_path, target_size=None,need=True):
    if target_size is None:
        target_size = [1024, 1024]

    def normalize(im, mean, std):
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        im = np.transpose(im, (2, 0, 1))
        return im

    im = cv2.imread(image_path)
    im = cv2.resize(im, target_size).astype('float32')
    print(im.shape)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if need:
        im = normalize(im, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return im


def save_imgs(results, backend_type, save_path=None):
    import os
    for i in range(results.shape[0]):
        result = get_pseudo_color_map(results[i])
        t_ls = os.path.basename(save_path).split(".")
        save_path = os.path.dirname(save_path) + "/" + t_ls[0] + "_" + backend_type + "." + t_ls[-1]
        print("save_path:{}".format(save_path))
        result.save(save_path)


def get_pseudo_color_map(pred, color_map=None):
    from PIL import Image as PILImage
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map

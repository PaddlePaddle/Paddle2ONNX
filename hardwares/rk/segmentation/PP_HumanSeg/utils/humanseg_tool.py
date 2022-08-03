import cv2
import numpy as np
def normalize(im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def preprocess(image, target_size=(192, 192)):
    image = normalize(image)
    image = resize(image, target_size=target_size)
    image = np.transpose(image, [2, 0, 1])
    # image = image[None, :, :, :]
    return image


def display_masked_image(mask, image, color_map=[255, 0, 0], weight=0.6):
    mask = mask > 0
    c1 = np.zeros(shape=mask.shape, dtype='uint8')
    c2 = np.zeros(shape=mask.shape, dtype='uint8')
    c3 = np.zeros(shape=mask.shape, dtype='uint8')
    pseudo_img = np.dstack((c1, c2, c3))
    for i in range(3):
        pseudo_img[:, :, i][mask] = color_map[i]
    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    return vis_result
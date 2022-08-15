import cv2
import numpy as np
import os

humanseg_std = [0.5, 0.5, 0.5]
humanseg_mean = [0.5, 0.5, 0.5]


def resize(im, target_size=192, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


class HumanSegPreProcess:
    def __init__(self,
                 target_size=None,
                 mean=None,
                 std=None):
        if mean is None:
            mean = humanseg_mean
        if target_size is None:
            target_size = (192, 192)
        if std is None:
            std = humanseg_std
        self.mean = mean
        self.std = std
        self.target_size = target_size

    def normalize(self, im):
        im = im.astype(np.float32, copy=False) / 255.0
        im -= self.mean
        im /= self.std
        return im

    def get_inputs(self, img, do_normalize=True):
        if isinstance(img, str):
            src_frame = cv2.imread(img)
        else:
            src_frame = img
        frame = cv2.cvtColor(src_frame, cv2.COLOR_BGRA2RGB)
        frame = resize(frame, self.target_size)
        if do_normalize:
            frame = self.normalize(frame)
            frame = frame.transpose((2, 0, 1))  # chw
        return frame, src_frame


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


class Humanseg:
    def __init__(self,
                 target_size=None):
        if target_size is None:
            target_size = (192, 192)
        self.target_size = target_size

    def predict(self, results, src_image, save_path, backend_type):
        print(np.array(results).shape)
        result = results[0][0]
        raw_frame = resize(src_image, self.target_size)
        pred = np.argmax(result, axis=0)

        image = display_masked_image(pred, raw_frame)
        image = resize(image, target_size=raw_frame.shape[0:2][::-1])
        t_ls = os.path.basename(save_path).split(".")
        save_path = os.path.dirname(save_path) + "/" + t_ls[0] + "_" + backend_type + "." + t_ls[-1]
        cv2.imwrite(save_path, image)

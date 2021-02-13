import numpy as np
import cv2
import centrist
import description as desc
from sklearn import preprocessing
import time

def normalized(rgb):

        norm=np.zeros(rgb.shape,np.float32)
        norm_rgb=np.zeros(rgb.shape,np.uint8)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        sum_rgb=b+g+r
        sum_rgb[sum_rgb == 0] = -1
        norm[:,:,0]=b/sum_rgb*255.0
        norm[:,:,1]=g/sum_rgb*255.0
        norm[:,:,2]=r/sum_rgb*255.0
        norm[norm < 0] = 0

        norm_rgb=cv2.convertScaleAbs(norm)
        norm_gr = cv2.cvtColor(norm_rgb, cv2.COLOR_BGR2GRAY)
        return norm_gr


def read_sensor_data(data, row_num):
    row = data[row_num].split(",")
    if row[1] == "[]":
        return None
    timestamp = float(row[0])
    temperature = float(row[1])
    humidity = float(row[2])
    pressure = float(row[3])
    if len(row) == 5:
        distance = float(row[4])
    else:
        distance = None
    return (timestamp, temperature, humidity, pressure, distance)


def temperature_decision_dirat(xpar, threshold):
    if len(xpar) < 40:
        return False
    x = np.array(xpar.copy())
    dx1 = np.mean(x[-39:] - x[:-1])
    #dx1 = np.mean(x[-39:] - x[-40:-1])
    dx2 = np.mean(x[-9:] - x[-10:-1])
    if (dx2 == 0):
        return 1
    ratio = np.abs((dx1+0.03)/np.float(dx2+0.03))
    if (ratio == 0):
        return 1
    r = np.max([ratio, 1/ratio])
    if r > threshold:
        return False
    return True


def temperature_decision(temperatures, stable_variance, alpha=1):

    if np.var(temperatures) > alpha * stable_variance:
        return False
    return True


def distance_decision(distance, last_distance, max_indoor_distance):
    start = time.time()
    if not (distance > max_indoor_distance) ^ (last_distance > max_indoor_distance):
        return True, time.time() - start
    else:
        return False, time.time() - start


def centrist_multiscale_desc(cl, im, bins=64):
    im1 = im.copy()
    im2 = cv2.resize(im1, dsize=(im1.shape[1]//2, im1.shape[0]//2))
    im3 = cv2.resize(im1, dsize=(im1.shape[1]//4, im1.shape[0]//4))
    im = centrist.centrist_im(cl, im)
    im2 = centrist.centrist_im(cl, im2)
    im3 = centrist.centrist_im(cl, im3)
    h1 = desc.spatial_histogram_bw(im3, 1, 1, 64)
    h2 = desc.spatial_histogram_bw(im2, 2, 2, 64)
    h3 = desc.spatial_histogram_bw(im, 4, 4, 64)
    desc_vector = np.concatenate((h1, h2, h3))
    desc_vector_sc = preprocessing.scale(desc_vector)
    return desc_vector_sc

def centrist_desc(cl, im):
    im1 = im[:,:,0]
    im2 = im[:,:,1]
    im3 = im[:,:,2]
    im1 = centrist.centrist_im(cl, im1)
    im2 = centrist.centrist_im(cl, im2)
    im3 = centrist.centrist_im(cl, im3)
    h1 = desc.spatial_histogram_bw(im1, 1, 1, 256)
    h2 = desc.spatial_histogram_bw(im2, 1, 1, 256)
    h3 = desc.spatial_histogram_bw(im3, 1, 1, 256)
    
    desc_vector = np.concatenate((h1, h2, h3))
    desc_vector_sc = preprocessing.scale(desc_vector)
    #print(len(h3))
    return desc_vector_sc


def spatial_histogram_hsv(image, w_cells, h_cells, bins):
    image = desc.bgr_to_hsv(image)
    w_step = image.shape[0]//w_cells
    h_step = image.shape[1]//h_cells
    sp_hist = np.array([])
    for w in range(w_cells):
        for h in range(h_cells):
            b = image[h * h_step:(h + 1)*h_step, w * w_step:(w + 1)*w_step, :]
            h1, _ = desc.histogram(b[:, :, 0], bins, 0, 179)
            h2, _ = desc.histogram(b[:, :, 1], bins, 0, 255)
            h3, _ = desc.histogram(b[:, :, 2], bins, 0, 255)
            h = np.concatenate((h1, h2, h3))
            sp_hist = np.concatenate((sp_hist, h))
    return sp_hist


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

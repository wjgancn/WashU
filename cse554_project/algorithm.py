from sklearn.linear_model import LinearRegression
import cv2 as cv
import numpy as np
import math


def FindLineAndLabel(img, target_x, target_y):

    thread_value = img[target_x - 5: target_x + 5, target_y - 5:target_y + 5].mean()

    # Binary Threshold
    _, img_bin = cv.threshold(img, thread_value, img.max(), cv.THRESH_BINARY)

    # Open Operation
    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)

    # Find connected components
    _, label = cv.connectedComponents(img_bin)

    # Remove other components
    label[label != label[target_x - 5: target_x + 5, target_y - 5:target_y + 5].max()] = 0

    # Linear Regression
    model = LinearRegression()
    x, y = label.nonzero()
    model.fit(x.reshape([x.shape[0], 1]), y)

    # Construct Line Data
    if -1 < model.coef_ < 1:
        t = np.linspace(0, 349, 350)
        y_ = t * model.coef_ + model.intercept_
    else:
        y_ = np.linspace(0, 349, 350)
        # t = y_ * model.coef_ + model.intercept_
        t = (y_ - model.intercept_)/model.coef_

    data_line = []
    for i in range(350):
        data_line.append(img[int(t[i]), int(y_[i])])

    # Find Two Node
    center_x = int(np.mean(x))

    data_line = np.array(data_line)
    data_line = np.convolve(data_line, np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]), 'same')

    node_l = 349
    node_r = 0
    for i in range(center_x, 350):
        if data_line[i] < data_line.mean():
            node_l = i
            break
    for i in range(0, center_x):
        if data_line[center_x - i] < data_line.mean():
            node_r = center_x - i
            break

    seg_line = []
    for i in range(node_r, node_l):
        seg_line.append(img[int(t[i]), int(y_[i])])
    seg_line = np.array(seg_line)

    if -1 < model.coef_ < 1:
        target_x = int((node_r + node_l) / 2)
        target_y = int(y_[target_x])
    else:
        target_y = int((node_r + node_l) / 2)
        target_x = int(t[target_y])

    thread_value = seg_line.mean() - 5

    # Binary Threshold
    _, img_bin = cv.threshold(img, thread_value, img.max(), cv.THRESH_BINARY)

    # Open Operation
    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)

    # Find connected components
    _, label = cv.connectedComponents(img_bin)

    # Remove other components
    label[label != label[target_x - 8: target_x + 8, target_y - 8:target_y + 8].max()] = 0

    model = LinearRegression()
    x, y = label.nonzero()
    model.fit(x.reshape([x.shape[0], 1]), y)

    return label, [model.coef_, model.intercept_], int((node_r + node_l)/2)


def UpdateLineAndLabel(img, line, centerpoint):
    # Construct Line Data
    if -1 < line[0] < 1:
        t = np.linspace(0, 349, 350)
        y_ = t * line[0] + line[1]
    else:
        y_ = np.linspace(0, 349, 350)
        # t = y_ * line[0] + line[1]
        t = (y_ - line[1])/line[0]

    data_line = []
    for i in range(350):
        data_line.append(img[int(t[i]), int(y_[i])])

    # Find Two Edege
    data_line = np.array(data_line)
    data_line = np.convolve(data_line, np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]), 'same')

    node_l = 349
    node_r = 0
    for i in range(centerpoint, 350):
        if data_line[i] < data_line.mean():
            node_l = i
            break
    for i in range(0, centerpoint):
        if data_line[centerpoint - i] < data_line.mean():
            node_r = centerpoint - i
            break

    seg_line = []
    for i in range(node_r, node_l):
        seg_line.append(img[int(t[i]), int(y_[i])])
    seg_line = np.array(seg_line)

    if -1 < line[0] < 1:
        target_x = int((node_r + node_l) / 2)
        target_y = int(y_[target_x])
    else:
        target_y = int((node_r + node_l) / 2)
        target_x = int(t[target_y])

    thread_value = seg_line.mean() - 5

    # Binary Threshold
    _, img_bin = cv.threshold(img, thread_value, img.max(), cv.THRESH_BINARY)

    # Open Operation
    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)

    # Find connected components
    _, label = cv.connectedComponents(img_bin)

    # Remove other components
    label[label != label[target_x - 3: target_x + 3, target_y - 3:target_y + 3].max()] = 0
    x, y = label.nonzero()

    y = y[np.logical_and(x < node_l, x > node_r)]
    x = x[np.logical_and(x < node_l, x > node_r)]

    dis = np.abs(x * line[0] - y + line[1])/np.sqrt(line[0]**2 + line[1]**2)
    x = x[dis < 0.05]
    y = y[dis < 0.05]

    label = np.zeros_like(label)
    for i in range(x.shape[0]):
        label[x[i], y[i]] = 1

    model = LinearRegression()
    model.fit(x.reshape([x.shape[0], 1]), y)

    return label, [model.coef_, model.intercept_], int((node_r + node_l) / 2),  \
           (node_l - node_r) * math.sqrt(1 + model.coef_**2)


import matplotlib.pyplot as plt
import numpy as np
import cv2


def double_threshold(image, lowThreshold, highThreshold, weak, isRatio=True):
    if isRatio:
        high = image.max() * highThreshold
        low = image.max() * lowThreshold
    else:
        high = highThreshold
        low = lowThreshold

    thresholdedImage = np.zeros(image.shape)

    strong = 255
    strongRow, strongColumn = np.where(image >= high)
    weakRow, weakColumn = np.where((image <= high) & (image >= low))

    thresholdedImage[strongRow, strongColumn] = strong
    thresholdedImage[weakRow, weakColumn] = weak

    return thresholdedImage


def initial_Threshold(source: np.ndarray):
    maxX = source.shape[1] - 1
    maxY = source.shape[0] - 1
    backMean = (
        int(source[0, 0])
        + int(source[0, maxX])
        + int(source[maxY, 0])
        + int(source[maxY, maxX])
    ) / 4
    sum = 0
    length = 0
    for i in range(0, source.shape[1]):
        for j in range(0, source.shape[0]):
            if not (
                (i == 0 and j == 0)
                or (i == maxX and j == 0)
                or (i == 0 and j == maxY)
                or (i == maxX and j == maxY)
            ):
                sum += source[j, i]
                length += 1
    foreMean = sum / length
    threshold = (backMean + foreMean) / 2
    return threshold


def optimal_Threshold(source: np.ndarray, threshold):
    back = source[np.where(source < threshold)]
    fore = source[np.where(source > threshold)]
    backMean = np.mean(back)
    foreMean = np.mean(fore)
    optimalThreshold = (backMean + foreMean) / 2
    return optimalThreshold


def global_optimal(source: np.ndarray):
    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    oldThreshold = initial_Threshold(src)
    newThreshold = optimal_Threshold(src, oldThreshold)
    iteration = 0
    while oldThreshold != newThreshold:
        oldThreshold = newThreshold
        newThreshold = optimal_Threshold(src, oldThreshold)
        iteration += 1
    return global_threshold(src, newThreshold)


def global_otsu(source: np.ndarray):
    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    yRange, xRange = src.shape
    histValues = plt.hist(src.ravel(), 256)[0]
    PDF = histValues / (yRange * xRange)
    CDF = np.cumsum(PDF)
    optimalThreshold = 1
    maxVariance = 0
    for t in range(1, 255):
        back = np.arange(0, t)
        fore = np.arange(t, 256)
        CDF2 = np.sum(PDF[t + 1 : 256])
        backMean = sum(back * PDF[0:t]) / CDF[t]
        foreMean = sum(fore * PDF[t:256]) / CDF2
        variance = CDF[t] * CDF2 * (foreMean - backMean) ** 2
        if variance > maxVariance:
            maxVariance = variance
            optimalThreshold = t
    return global_threshold(src, optimalThreshold)


def global_spectral(source: np.ndarray):
    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    yRange, xRange = src.shape
    histValues = plt.hist(src.ravel(), 256)[0]
    PDF = histValues / (yRange * xRange)
    CDF = np.cumsum(PDF)
    optimallow = 1
    optimalhigh = 1
    maxVariance = 0
    Global = np.arange(0, 256)
    gMean = sum(Global * PDF) / CDF[-1]
    for lowT in range(1, 254):
        for highT in range(lowT + 1, 255):
            try:
                back = np.arange(0, lowT)
                low = np.arange(lowT, highT)
                high = np.arange(highT, 256)
                CDFL = np.sum(PDF[lowT:highT])
                CDFH = np.sum(PDF[highT:256])
                backMean = sum(back * PDF[0:lowT]) / CDF[lowT]
                lowMean = sum(low * PDF[lowT:highT]) / CDFL
                highMean = sum(high * PDF[highT:256]) / CDFH
                variance = (
                    CDF[lowT] * (backMean - gMean) ** 2
                    + (CDFL * (lowMean - gMean) ** 2)
                    + (CDFH * (highMean - gMean) ** 2)
                )
                if variance > maxVariance:
                    maxVariance = variance
                    optimallow = lowT
                    optimalhigh = highT
            except RuntimeWarning:
                pass
    return double_threshold(src, optimallow, optimalhigh, 128, False)


def global_threshold(source: np.ndarray, threshold: int):
    src = np.copy(source)
    row, column = src.shape
    for x in range(column):
        for y in range(row):
            if src[x, y] > threshold:
                src[x, y] = 1
            else:
                src[x, y] = 0

    return src


def local_threshold(source: np.ndarray, regions, thresholdingFunction):
    src = np.copy(source)
    yMax, xMax = src.shape
    result = np.zeros((yMax, xMax))
    yStep = yMax // regions
    xStep = xMax // regions
    xRange = []
    yRange = []
    for i in range(0, regions + 1):
        xRange.append(xStep * i)

    for i in range(0, regions + 1):
        yRange.append(yStep * i)

    for x in range(0, regions):
        for y in range(0, regions):
            result[
                yRange[y] : yRange[y + 1], xRange[x] : xRange[x + 1]
            ] = thresholdingFunction(
                src[yRange[y] : yRange[y + 1], xRange[x] : xRange[x + 1]]
            )
    return result


def local_otsu(source):
    return local_threshold(source, 4, global_otsu)


def local_optimal(source):
    return local_threshold(source, 4, global_optimal)


def local_spectral(source):
    return local_threshold(source, 4, global_spectral)

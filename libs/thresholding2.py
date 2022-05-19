import matplotlib.pyplot as plt
import numpy as np


def DoubleThreshold(Image, LowThreshold, HighThreshold, Weak, isRatio=True):

    if isRatio:
        High = Image.max() * HighThreshold
        Low = Image.max() * LowThreshold
    else:
        High = HighThreshold
        Low = LowThreshold

    ThresholdedImage = np.zeros(Image.shape)

    Strong = 255
    StrongRow, StrongCol = np.where(Image >= High)
    WeakRow, WeakCol = np.where((Image <= High) & (Image >= Low))

    ThresholdedImage[StrongRow, StrongCol] = Strong
    ThresholdedImage[WeakRow, WeakCol] = Weak

    return ThresholdedImage




def Initial_Threshold(source: np.ndarray):
    MaxX = source.shape[1] - 1
    MaxY = source.shape[0] - 1
    BackMean = (int(source[0, 0]) + int(source[0, MaxX]) + int(source[MaxY, 0]) + int(source[MaxY, MaxX])) / 4
    Sum = 0
    Length = 0
    for i in range(0, source.shape[1]):
        for j in range(0, source.shape[0]):
            if not ((i == 0 and j == 0) or (i == MaxX and j == 0) or (i == 0 and j == MaxY) or (
                    i == MaxX and j == MaxY)):
                Sum += source[j, i]
                Length += 1
    ForeMean = Sum / Length
    Threshold = (BackMean + ForeMean) / 2
    return Threshold


def Optimal_Threshold(source: np.ndarray, Threshold):
 
    Back = source[np.where(source < Threshold)]
    Fore = source[np.where(source > Threshold)]
    BackMean = np.mean(Back)
    ForeMean = np.mean(Fore)
    OptimalThreshold = (BackMean + ForeMean) / 2
    return OptimalThreshold

def optimal(source: np.ndarray):
    src = np.copy(source)
    OldThreshold = Initial_Threshold(src)
    NewThreshold = Optimal_Threshold(src, OldThreshold)
    iteration = 0
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = Optimal_Threshold(src, OldThreshold)
        iteration += 1
    
    return global_threshold(src, NewThreshold)


def otsu(source: np.ndarray):
    src = np.copy(source)
    YRange, XRange = src.shape
    HistValues = plt.hist(src.ravel(), 256)[0]
    PDF = HistValues / (YRange * XRange)
    CDF = np.cumsum(PDF)
    OptimalThreshold = 1
    MaxVariance = 0
    for t in range(1, 255):
        Back = np.arange(0, t)
        Fore = np.arange(t, 256)
        CDF2 = np.sum(PDF[t + 1:256])
        BackMean = sum(Back * PDF[0:t]) / CDF[t]
        ForeMean = sum(Fore * PDF[t:256]) / CDF2
        Variance = CDF[t] * CDF2 * (ForeMean - BackMean) ** 2
        if Variance > MaxVariance:
            MaxVariance = Variance
            OptimalThreshold = t
    return global_threshold(src, OptimalThreshold)


def spectral(source: np.ndarray):


    src = np.copy(source)
    # Get Image Dimensions
    YRange, XRange = src.shape
    HistValues = plt.hist(src.ravel(), 256)[0]
    PDF = HistValues / (YRange * XRange)
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            try:
                Back = np.arange(0, LowT)
                Low = np.arange(LowT, HighT)
                High = np.arange(HighT, 256)
                CDFL = np.sum(PDF[LowT:HighT])
                CDFH = np.sum(PDF[HighT:256])
                BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                HighMean = sum(High * PDF[HighT:256]) / CDFH
                Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                        CDFH * (HighMean - GMean) ** 2))
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowT
                    OptimalHigh = HighT
            except RuntimeWarning:
                pass
    return DoubleThreshold(src, OptimalLow, OptimalHigh, 128, False)



def global_threshold(source: np.ndarray, threshold: int):
    src = np.copy(source)
    row, column = src.shape
    for x in range(column):
        for y in range(row):
         if src[x,y] > threshold:
             src[x,y] = 1
         else:
             src[x,y] = 0

    return src


def LocalThresholding(source: np.ndarray, Regions, ThresholdingFunction):
    
    src = np.copy(source)
    YMax, XMax = src.shape
    Result = np.zeros((YMax, XMax))
    YStep = YMax // Regions
    XStep = XMax // Regions
    XRange = []
    YRange = []
    for i in range(0, Regions+1):
        XRange.append(XStep * i)

    for i in range(0, Regions+1):
        YRange.append(YStep * i)

    
    for x in range(0, Regions):
        for y in range(0, Regions):
            Result[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = ThresholdingFunction(src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]])
    return Result







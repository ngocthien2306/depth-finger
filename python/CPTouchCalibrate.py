import cv2
import numpy as np
import math

screenResolutionX = 1280 # Projector resolution X
screenResolutionY = 720 # Projector resolution Y
screenMatrix = [(0, 0), (0, screenResolutionY), (screenResolutionX, screenResolutionY), (screenResolutionX, 0)]
offsetX = 1920 * 1 # Second screen of projector when extension e.g. Main screen resolution(1920*1080)
defScalbArea1 = 100000 # 40cm: 115000 # 68cm: 100000
defScalbArea2 = 100000 # 40cm: 115000 # 68cm: 100000

class CPTouchCalibrate():

  def __init__(self) -> None:
    self.alreadyCalibrate = False
    self.calibrateMatrix = None
    self.imageList = []

  def _getCalibrateMartix(self, srcMatrix):
    print(f" ************************** Get CalibrateMatrix **************************")
    srcNumpyMatrix = np.array(srcMatrix, dtype=np.float32)
    dstNumpyMatrix = np.array(screenMatrix, dtype=np.float32)
    print(srcNumpyMatrix)
    print(dstNumpyMatrix)
    M = cv2.getPerspectiveTransform(srcNumpyMatrix, dstNumpyMatrix)
    print(f"M:{M}")
    print(f" ************************** End CalibrateMatrix **************************")
    return M

  def convertPoint(self, srcPosition):
    dstPosition = srcPosition
    if dstPosition != (0, 0, 0, -1):
      srcPoint = (srcPosition[1], srcPosition[2])
      if self.alreadyCalibrate == True:
         srcPoint = np.array([*srcPoint, 1])
         w = np.matmul(self.calibrateMatrix, srcPoint)
         cx = min(max(round(w[0] / w[2]), 0), screenResolutionX - 1) + offsetX
         cy = min(max(round(w[1] / w[2]), 0), screenResolutionY - 1)
         dstPosition = (srcPosition[0], cx, cy, srcPosition[3])
    return dstPosition

  def startMapping(self, frame):
    self.alreadyCalibrate = False
    self.imageList.append(frame)
    area, x, y, w, h = 0, 0, 0, 0, 0
    result = -1
    if len(self.imageList) >= 50:
        accumulatedImage = None
        for img in self.imageList:
            if accumulatedImage is None:
                accumulatedImage = np.zeros_like(img, dtype=np.float32)
            accumulatedImage += img
        accumulatedImage = (accumulatedImage / len(self.imageList)).astype(np.uint8)
        thresholdValue = 70
        _, accumulatedImage = cv2.threshold(accumulatedImage, thresholdValue, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        accumulatedImage = cv2.erode(accumulatedImage, kernel) 
        accumulatedImage = cv2.dilate(accumulatedImage, kernel) 
        gray = cv2.cvtColor(accumulatedImage, cv2.COLOR_BGR2GRAY)

        # Find conters
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ix, iy, iw, ih = cv2.boundingRect(contour)
            if area < iw * ih and iw * ih > defScalbArea1:
                area = iw * ih
                x, y, w, h = ix, iy, iw, ih
        self.imageList.pop(0)
    if area != 0:
        p1, p2, p3, p4 = (x, y), (x+w, y), (x+w, y+h), (x, y+h)
        srcMatrix = p2, p1, p4, p3
        self.calibrateMatrix = self._getCalibrateMartix(srcMatrix)
        self.imageList = []
        result = 0
        self.alreadyCalibrate = True
        drawImg = frame.copy()
        cv2.circle(drawImg, p1, 3, (255, 0, 0), 3)
        cv2.circle(drawImg, p2, 3, (0, 255, 0), 3)
        cv2.circle(drawImg, p3, 3, (0, 0, 255), 3)
        cv2.circle(drawImg, p4, 3, (255, 255, 0), 3)
        cv2.imshow("Result", drawImg)
        cv2.waitKey(1000)
        cv2.destroyWindow("Result")
    return result, x, y, w, h
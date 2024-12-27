from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class KeypointTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()  # Mapping object ID -> keypoints
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxObjects = 2  # Limit number of tracked objects

    def register(self, keypoints):
        # Register an object with its keypoints
        if len(self.objects) >= self.maxObjects:
            # Remove the least recently updated object
            oldestID = next(iter(self.objects))
            self.deregister(oldestID)
        self.objects[self.nextObjectID] = keypoints
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID = (self.nextObjectID + 1) % self.maxObjects  # Cycle IDs between 0 and 1

    def deregister(self, objectID):
        # Deregister an object ID
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, keypoints_list):
        # Check if the list of input keypoints is empty
        if len(keypoints_list) == 0:
            # Reset object tracking and ID counter
            self.objects.clear()
            self.disappeared.clear()
            self.nextObjectID = 0
            return self.objects

        # If no objects are being tracked, register all keypoints
        if len(self.objects) == 0:
            for keypoints in keypoints_list[:self.maxObjects]:  # Limit to maxObjects
                self.register(keypoints)
        else:
            # Grab the set of object IDs and their keypoints
            objectIDs = list(self.objects.keys())
            objectKeypoints = list(self.objects.values())

            # Compute distance matrix between object keypoints and input keypoints
            D = np.zeros((len(objectKeypoints), len(keypoints_list)))

            for i, obj_kps in enumerate(objectKeypoints):
                for j, inp_kps in enumerate(keypoints_list):
                    # Calculate mean distance between corresponding keypoints
                    D[i, j] = np.mean([dist.euclidean(obj_kps[k], inp_kps[k]) for k in range(len(obj_kps))])

            # Find the smallest value in each row and sort the row indexes
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = keypoints_list[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(keypoints_list[col])

        return self.objects


if __name__ == "__main__":
    # Example usage
    tracker = KeypointTracker()

    # Example input: each item is a list of keypoints (x, y) for an object
    frame1 = [
        [(10, 20), (15, 25), (20, 30)],  # Keypoints for object 1
        [(100, 200), (110, 210), (120, 220)]  # Keypoints for object 2
    ]

    frame2 = [
        [(12, 22), (17, 27), (22, 32)],  # Slightly moved object 1
        [(105, 205), (115, 215), (125, 225)]  # Slightly moved object 2
    ]

    frame3 = [
        [(50, 60), (55, 65), (60, 70)],  # New object
        [(150, 160), (155, 165), (160, 170)]  # Another new object
    ]

    print("Frame 1:")
    print(tracker.update(frame1))

    print("\nFrame 2:")
    print(tracker.update(frame2))

    print("\nFrame 3 (Replace Old Objects):")
    print(tracker.update(frame3))

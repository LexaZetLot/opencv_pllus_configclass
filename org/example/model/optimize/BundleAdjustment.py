import cv2
import numpy as np


class BundleAdjustment:
    def _get3dPos(self, pos, ob, r, t, K):
        """оптимизация ошибки"""
        p, J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        if abs(e[0]) > 0.5 or abs(e[1]) > 1:
            return None
        return pos

    def bundleAdjustment(self, rotations, motions, K, correspondStructIdx, keyPointsForAll, structure):
        """минимизация ошибки"""
        for i in range(len(rotations)):
            r, _ = cv2.Rodrigues(rotations[i])
            rotations[i] = r

        for i in range(len(correspondStructIdx)):
            point3dIds = correspondStructIdx[i]
            keyPoints = keyPointsForAll[i]
            r = rotations[i]
            t = motions[i]

            for j in range(len(point3dIds)):
                point3dId = int(point3dIds[j])
                if point3dId < 0:
                    continue
                newPoint = self._get3dPos(structure[point3dId], keyPoints[j].pt, r, t, K)
                structure[point3dId] = newPoint

        return structure
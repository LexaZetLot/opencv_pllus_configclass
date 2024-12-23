import os
import cv2
import math
import numpy as np

from burp_spring.config.annotation.Autowired import Autowired
from org.example.service.ServiceImpl import ServiceImpl


class SFM(ServiceImpl):
    def __init__(self, dirImageFile='../../../data_photo/',
                       K=np.array([[997.6878623828,   0.0000000000, 474.1075945276],
                                   [  0.0000000000, 997.8994565544, 634.6883539108],
                                   [  0.0000000000,   0.0000000000,   1.0000000000]])):
        self.K = K
        self.structure = None
        self.bundleAdjustment = None
        self.PLY = None
        self.dirImageFile = dirImageFile

    @Autowired()
    def getBundleAdjustment(self, bundleAdjustment):
        self.bundleAdjustment = bundleAdjustment

    @Autowired()
    def getPLY(self, PLY):
        self.PLY = PLY

    def run(self):
        imgNames = os.listdir(os.path.abspath(__file__)[:len(os.path.abspath(__file__)) - 6] + self.dirImageFile)
        imgNames = sorted(imgNames)
        for i in range(len(imgNames)):
            imgNames[i] = os.path.abspath(__file__)[:len(os.path.abspath(__file__)) - 6] + self.dirImageFile + imgNames[i]

        keyPointsForAll, descriptorForAll = self._extractFeatures(imgNames)
        matchesForAll = self._matchAllFeatures(descriptorForAll)
        correspondStructIdx, rotations, motions = self._initStructure(self.K, keyPointsForAll, matchesForAll)


        for i in range(1, len(matchesForAll)):
            objectPoints, imagePoints = self._getObjPointsAndImgPoints(matchesForAll[i], correspondStructIdx[i],
                                                                       keyPointsForAll[i + 1])

            if len(imagePoints) < 7:
                while len(imagePoints) < 7:
                    objectPoints = np.append(objectPoints, [objectPoints[0]], axis=0)
                    imagePoints = np.append(imagePoints, [imagePoints[0]], axis=0)

            _, r, T, _ = cv2.solvePnPRansac(objectPoints, imagePoints, self.K, np.array([]))
            R, _ = cv2.Rodrigues(r)
            rotations.append(R)
            motions.append(T)
            p1, p2 = self._getMatchedPoints(keyPointsForAll[i], keyPointsForAll[i + 1], matchesForAll[i])
            nextStructure = self._reconstruct(self.K, rotations[i], motions[i], R, T, p1, p2)

            correspondStructIdx[i], correspondStructIdx[i + 1] = self._fusionStructure(matchesForAll[i],
                                                                                       correspondStructIdx[i],
                                                                                       correspondStructIdx[i + 1],
                                                                                       nextStructure)


        self.structure = self.bundleAdjustment.bundleAdjustment(rotations,
                                                                motions,
                                                                self.K,
                                                                correspondStructIdx,
                                                                keyPointsForAll,
                                                                self.structure)

        self.structure = self.structure[~np.isnan(self.structure[:, 0])]
        self.PLY.save(self.structure)


    def _extractFeatures(self, imageNames):
        """
        ищет в изображении по адрессу фотки точки интереса и дискрипторы а также их цвет после чего возвращает в виде
        массива
        image_names: полные имена фото
        """
        sift = cv2.SIFT.create(0, 3, 0.04, 10)
        keyPointsForAll = []
        descriptorForAll = []

        for image_name in imageNames:
            image = cv2.imread(image_name)

            if image is None:
                continue
            kep, des = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

            if len(kep) <= 10:
                continue

            keyPointsForAll.append(kep)
            descriptorForAll.append(des)

        return np.array(keyPointsForAll, dtype=object), np.array(descriptorForAll, dtype=object)


    def _matchFeatures(self, query, train):
        """ищет матчи и отбирает лучшие"""
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knnMatches = bf.knnMatch(query, train, k=2)
        matches = []

        for m, n in knnMatches:
            if m.distance < 0.75 * n.distance:
                matches.append(m)

        return np.array(matches)

    def _matchAllFeatures(self, descriptorForAll):
        """матчи последовательности"""

        matchesForAll = []
        for i in range(len(descriptorForAll) - 1):
            matches = self._matchFeatures(descriptorForAll[i], descriptorForAll[i + 1])
            matchesForAll.append(matches)
        return np.array(matchesForAll, dtype=object)

    def _findTransform(self, K, p1, p2):
        """основная матрица + востоновление матрцы поворота и вектора смещения"""
        focalLength = 0.5 * (K[0, 0] + K[1, 1])
        principlePoint = (K[0, 2], K[1, 2])

        E, mask = cv2.findEssentialMat(p1, p2, focalLength, principlePoint, cv2.RANSAC, 0.999, 1.0)
        cameraMatrix = np.array([[focalLength, 0, principlePoint[0]], [0, focalLength, principlePoint[1]], [0, 0, 1]])
        _, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)

        return R, T, mask

    def _getMatchedPoints(self, p1, p2, matches):
        """точки в матчах"""
        pts1 = np.asarray([p1[m.queryIdx].pt for m in matches])
        pts2 = np.asarray([p2[m.trainIdx].pt for m in matches])

        return pts1, pts2

    def _maskoutPoints(self, p1, mask):
        """приминяют маску к массиву"""
        p1_copy = []
        for i in range(len(mask)):
            if mask[i] > 0:
                p1_copy.append(p1[i])

        return np.array(p1_copy)

    def _initStructure(self, K, keyPointsForAll, matchesForAll):
        """вотсановление траингуляция соответвия 2D + 3D"""
        p1, p2 = self._getMatchedPoints(keyPointsForAll[0], keyPointsForAll[1], matchesForAll[0])


        if self._findTransform(K, p1, p2):
            R, T, mask = self._findTransform(K, p1, p2)
        else:
            R, T, mask = np.array([]), np.array([]), np.array([])

        p1 = self._maskoutPoints(p1, mask)
        p2 = self._maskoutPoints(p2, mask)

        R0 = np.eye(3, 3)
        T0 = np.zeros((3, 1))
        self.structure = self._reconstruct(K, R0, T0, R, T, p1, p2)
        rotations = [R0, R]
        motions = [T0, T]

        correspondStructIdx = []
        for key_p in keyPointsForAll:
            correspondStructIdx.append(np.ones(len(key_p)) * -1)
        correspondStructIdx = np.array(correspondStructIdx, dtype=object)

        idx = 0
        matches = matchesForAll[0]
        for i, match in enumerate(matches):
            if mask[i] == 0:
                continue
            correspondStructIdx[0][int(match.queryIdx)] = idx
            correspondStructIdx[1][int(match.trainIdx)] = idx
            idx += 1

        return correspondStructIdx, rotations, motions

    def _reconstruct(self, K, R1, T1, R2, T2, p1, p2):
        """траингуляция + гомогенные в эвклидовые"""
        proj1 = np.zeros((3, 4))
        proj2 = np.zeros((3, 4))
        proj1[0:3, 0:3] = np.float32(R1)
        proj1[:, 3] = np.float32(T1.T)
        proj2[0:3, 0:3] = np.float32(R2)
        proj2[:, 3] = np.float32(T2.T)
        fk = np.float32(K)
        proj1 = np.dot(fk, proj1)
        proj2 = np.dot(fk, proj2)
        s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)

        structure = []
        for i in range(len(s[0])):
            col = s[:, i]
            col /= col[3]
            structure.append([col[0], col[1], col[2]])

        return np.array(structure)

    def _fusionStructure(self, matches, structIndices, nextStructIndices, nextStructure):
        """
        Если совпадающие точки на предыдущем изображении уже связаны с 3D-точкой, то текущая точка связывается с той же 3D-точкой.
        Если совпадения новой точки не существует в текущей структуре, она добавляется в массив structure, и индексы обновляются.
        """
        for i, match in enumerate(matches):
            queryIdx = match.queryIdx
            trainIdx = match.trainIdx
            structIdx = structIndices[queryIdx]
            if structIdx >= 0:
                nextStructIndices[trainIdx] = structIdx
                continue
            self.structure = np.append(self.structure, [nextStructure[i]], axis=0)
            structIndices[queryIdx] = nextStructIndices[trainIdx] = len(self.structure) - 1
        return structIndices, nextStructIndices

    def _getObjPointsAndImgPoints(self, matches, structIndices, keyPoints):
        """
        Функция собирает пары:
        Объектные точки (3D): Координаты из реконструированной структуры.
        Изображенческие точки (2D): Координаты соответствующих ключевых точек на текущем изображении.
        """
        objectPoints = []
        imagePoints = []
        for match in matches:
            queryIdx = match.queryIdx
            trainIdx = match.trainIdx
            structIdx = structIndices[queryIdx]
            if structIdx < 0:
                continue
            objectPoints.append(self.structure[int(structIdx)])
            imagePoints.append(keyPoints[trainIdx].pt)

        return np.array(objectPoints), np.array(imagePoints)


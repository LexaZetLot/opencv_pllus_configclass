import os
import random
import textwrap

import pymongo
import cv2
import base64
import numpy as np


class ImgDAO(object):
    def __init__(self, url = 'mongodb://root:password@172.30.0.3:27017', client = 'ImgDataBase'):
        self.client = pymongo.MongoClient(url)
        self.base = self.client[client]

    def InsertImgList(self, collectionName , imgList):
        collection = self.base[collectionName]
        encodeList = []
        for img in imgList:
            _, buffer = cv2.imencode('.pgm', img)
            encodeList.append(base64.b64encode(buffer))
        collection.insert_one({"_id": 1, "imgList": encodeList[0:101]})
        collection.insert_one({"_id": 2, "imgList": encodeList[101:201]})
        collection.insert_one({"_id": 3, "imgList": encodeList[201:]})

    def ReadImgList(self):
        labels = 0
        imgResult = []
        labelsResult = []
        names = []

        for collectionName in self.base.list_collection_names():
            if collectionName != '__fileModelSave__':
                collection = self.base[collectionName]
                names.append(collectionName)
                result = collection.find()

                for res in result:
                    for img in res['imgList']:
                        jpg_original = base64.b64decode(img)
                        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                        imgResult.append(cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_GRAYSCALE))
                        labelsResult.append(labels)
                labels += 1

        imgResultNumpy = np.array(imgResult)
        labelsResultNumpy = np.array(labelsResult)
        return names, imgResultNumpy, labelsResultNumpy

    def NameList(self):
        restList = []
        for name in self.base.list_collection_names():
            if name != '__fileModelSave__':
                restList.append(name)
        return restList

    def InsertXmlSetting(self, path):
        collection = self.base['__fileModelSave__']
        collection.delete_many({})

        fp = open(path, 'r')
        string = fp.read()
        string = string.encode('ascii')
        string = base64.b64encode(string)
        string = string.decode('ascii')
        listString = textwrap.wrap(string, 16700000)

        idCount = 1
        for string in listString:
            collection.insert_one({"_id": idCount, "xmlSetting": string})
            idCount += 1

        os.remove(path)

    def ReadXmlSetting(self) -> str:
        collection = self.base['__fileModelSave__']
        path = './' + str(random.randrange(1000000)) + '.xml'
        while os.path.exists(path):
            path = './' + str(random.randrange(1000000)) + '.xml'

        fp = open(path, 'w')
        result = collection.find()
        string = ''

        for res in result:
            string = string + res['xmlSetting']

        string = string.encode('ascii')
        string = base64.b64decode(string)
        string = string.decode('ascii')

        fp.write(string)
        return path
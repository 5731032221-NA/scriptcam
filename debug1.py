
import numpy as np
import cv2
import os
import uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pymongo
from Crypto.Cipher import AES
from Crypto import Random
import base64
# import io
from io import BytesIO
from PIL import Image
from pkcs7 import PKCS7Encoder
# from io import StringIO
# import sys
# import os

# sys.stdout = open(os.devnull, "w")

# import asyncio

# import concurrent.futures
import _thread

import requests
import json
from datetime import  timedelta, datetime, date, time as t2
import time

now=datetime.now()

current_time=now.strftime("%H%M%S")

blob_service_client = BlobServiceClient.from_connection_string(
    "DefaultEndpointsProtocol=https;AccountName=oneteamblob;AccountKey=qcv7bSwg5vFNZRt1gY9XLPcv6OWKdKakKCj5znpUQRNQTPAOkLbhnCuZpt/1m4Gc9f5tV55x0CEzcVWjCubTaQ==;EndpointSuffix=core.windows.net")

# Create a unique name for the container
container_name = "facedetection"
cap = cv2.VideoCapture("rtsp://admin:admin@10.76.53.14:8554/stream0/out.h264")


def storeblob(name):
    print(name)
    blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=name)
    with open("datadebug1/"+name, "rb") as data:
        blob_client.upload_blob(data) 


def imagescan(frame, count):
    # print("cc",count)
    if (count % 36) == 0:
        now=datetime.now() + timedelta(hours=7)
        today=date.today() + timedelta(hours=7)
        current_time=now.strftime("%H%M%S")
        name="test"+str(today)+"-1-"+current_time+".jpg"
        print("count",count)
        cv2.imwrite("datadebug1/"+name, frame)
        storeblob(name)

count1=1
while(True):
    ret, img=cap.read()
    timenow =datetime.now() + timedelta(hours=7)
    # bool1 = ((int(t2(5,30).strftime("%H%M"))<int( (timenow).strftime("%H%M")) ) & (int(t2(8,00).strftime("%H%M"))>int( (timenow).strftime("%H%M")) )  ) & ((timenow).weekday() < 5)
    if ((cv2.waitKey(20) & 0xFF == ord('q'))):
        break

    if ret:
        _thread.start_new_thread(imagescan, (img, count1))
    else: 
        break
    count1=count1 + 1


cap.release()
cv2.destroyAllWindows()
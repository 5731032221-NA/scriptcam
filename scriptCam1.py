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
from random import randint
# set to your own subscription key value
subscription_key = '8b1838e13407455daf92a98bd51016ba'
# key = Random.new().read(AES.block_size)
# iv = Random.new().read(AES.block_size)
# import sys
# sys.stderr = object

# from cryptemis import Cryptemis
# KEEP_FILENAME = True
# PASSWORD = 'my_super_password'
# ENCRYPT = True
# DECRYPT = False
# import pyAesCrypt
# from os import stat, remove
# bufferSize = 64 * 1024
# password = "foopassword"
# try:
    ##print("Azure Blob storage v12 - Python quickstart sample")
    # Quick start code goes here
# except Exception as ex:
    ##print('Exception:')
    ##print(ex)

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(
    "DefaultEndpointsProtocol=https;AccountName=oneteamblob;AccountKey=qcv7bSwg5vFNZRt1gY9XLPcv6OWKdKakKCj5znpUQRNQTPAOkLbhnCuZpt/1m4Gc9f5tV55x0CEzcVWjCubTaQ==;EndpointSuffix=core.windows.net")

# Create a unique name for the container
container_name = "facedetection"

# Create the container
# container_client = blob_service_client.create_container(container_name)

# Create a file in local Documents directory to upload and download
# local_path = "./data"
# local_file_name = "quickstart" + str(uuid.uuid4()) + ".txt"
# upload_file_path = os.path.join(local_path, local_file_name)

# Write text to the file
# file = open(upload_file_path, 'w')
# file.write("Hello, World!")
# file.close()

# Create a blob client using the local file name as the name for the blob
# blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

# Upload the created file
# with open(upload_file_path, "rb") as data:
# blob_client.upload_blob(data)

# import requests
# url = 'http://1teamapi.azurewebsites.net/upload'
# headers = {'Content-type': 'multipart/form-data', 'Accept': 'application/json'}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(   "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
# cap = cv2.VideoCapture("20200108v2.mp4")
cap = cv2.VideoCapture("http://10.76.53.14:8090/video")



def resize(img):
    scale_percent = 80  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def storeblob(name):
    print(name)
    blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=name)
    with open("data/"+name, "rb") as data:
        blob_client.upload_blob(data) 

def encrypt_val(clear_text):
    master_key = b'meafacialkeycam1' 
    encoder = PKCS7Encoder()
    raw = encoder.encode(clear_text)
    iv = Random.new().read( 16 )
    cipher = AES.new( master_key, AES.MODE_CBC, iv, segment_size=128 )
    return base64.b64encode( iv + cipher.encrypt( raw ) ) 

def storecrop(name):
    jpgfile = Image.open("data/"+name)
    buffered = BytesIO()
    jpgfile.save(buffered, format="JPEG")
    img_b = base64.b64encode(buffered.getvalue())
    encoding = 'utf-8'
    img_str = str(img_b, encoding)
    img_enc = encrypt_val(img_str)
    img_enc_str = str(img_enc, encoding)

    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    db = client.image
    db.crop.insert_one({
        "name": name,
        "data": img_enc_str
    }
    )
    





def apiidentify(name, arrfaceid):

    identify_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/identify'
    header = {'Ocp-Apim-Subscription-Key': subscription_key}
    paramsIden = json.loads('{ "personGroupId": "oneteam", "faceIds": '+json.dumps(
        arrfaceid)+',"confidenceThreshold": 0.1, "maxNumOfCandidatesReturned": 1 }')
    return requests.post(identify_url,  headers=header, json=paramsIden)


def apidetect(name):
    face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

    image_url = "https://oneteamblob.blob.core.windows.net/facedetection/"+name
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'true',
        'returnFaceAttributes': 'emotion,gender,age,blur',
    }
    return requests.post(
        face_api_url, params=params, headers=headers, json={"url": image_url})

def getemo(emotion):
    maxProp = ''
    maxValue = -1
    secProp = ''
    secValue = -1
    for prop in emotion:
        value = emotion[prop]
        if (value > maxValue):
            secValue = maxValue
            secProp = maxProp
            maxValue = value
            maxProp = prop
        elif (secValue < value):
            secValue = value
            secProp = prop
           
    emostr = 'neutral'
    if (maxProp == 'neutral'):
        if (maxValue == 1):
            emostr = 'neutral'
        else:
            if (secProp == 'anger'):
                emostr = 'anger'
            elif (secProp == 'contempt'):
                emostr = 'contempt'
            elif (secProp == 'disgust'): 
                emostr = 'disgust'
            elif (secProp == 'fear'):
                emostr = 'fear'
            elif (secProp == 'happiness'): 
                emostr = 'happiness'
            elif (secProp == 'sadness'):
                emostr = 'sadness'
            elif (secProp == 'surprise'):
                emostr = 'surprise'
    else:
        if (maxProp == 'anger'):
            emostr = 'anger'
        elif (maxProp == 'contempt'):
            emostr = 'contempt'
        elif (maxProp == 'disgust'):
            emostr = 'disgust'
        elif (maxProp == 'fear'):
            emostr = 'fear'
        elif (maxProp == 'happiness'):
            emostr = 'happiness'
        elif (maxProp == 'sadness'):
            emostr = 'sadness'
        elif (maxProp == 'surprise'):
            emostr = 'surprise'
    return emostr

def mongo(now,timei, nameperson, checkin, faceAttributes, faceRectangle, image_url, imageCropUrl):
    today = date.today()
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    db = client.checkin
    emo = getemo(faceAttributes['emotion'])
    query = {"name": nameperson}
    queryy = db.checkin[today].find(query)
    if(queryy.count() > 0):
        newvalues = { "$set": { "checkout": checkin ,"checkoutEmotion": faceAttributes} }

    else:
        db.checkin[today].update(
        query,
        {
            "name": nameperson,
            "checkin": checkin,
            "checkindatetime": now.strftime("%Y%m%d%H%M%S"),
            "checkinMonth": today.strftime("%Y-%m"),
            "checkinEmotion": faceAttributes,
            "checkinEmo": emo,
            "checkinImageCrop": imageCropUrl,
            "camerain": 1,
            "checkout": "",
            "checkoutEmotion": {"gender":"","age":0},
            "checkoutEmo": "",
            "checkoutImageCrop": "",
            "cameraout": 0,
            "checkoutdatetime": "",
            "checkoutMonth":""
        },
            upsert=True
        )   
        db.checkattendance.insert_one({
        "name": nameperson,
        "checkin": { "time": now.strftime("%H:%M:%S"),
                     "emotion": faceAttributes
        },
        "checkout": { "time": "",
                     "emotion": ""
        },
        "Date":  now.strftime("%Y-%m-%d"),
        "faceAttributes": faceAttributes
        }
        )


   

    


def imagescan(img, count):
    if (count % 1000) == 0:
        print("count",count)
        #time.sleep(count/60)
        # frame=resize(img)
        frame = img
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, 1.1, 4)
        if(len(faces) > 0):
            now=datetime.now() + timedelta(hours=7)
            today=date.today()
            current_time=now.strftime("%H%M%S")
            name=str(today)+"1-"+current_time+str(randint(0, 100))+".jpg"
            cv2.imwrite("data/"+name, frame)

            storeblob(name)

            response=apidetect(name)
            detect=response.json()
            
            if(detect != []):
                arrfaceid=[]
                for face in detect:
                    arrfaceid.append(face[u'faceId'])
                response=apiidentify(name, arrfaceid)
                identify=response.json()
                for index, iden in enumerate(identify):
                    uriPerson='https://southeastasia.api.cognitive.microsoft.com/face/v1.0/persongroups/oneteam/persons/' + \
                        str(json.dumps(identify[0][u'candidates'][0][u'personId'])).replace(
                            '"', '')
                    header={'Ocp-Apim-Subscription-Key': subscription_key}
                    crop_img=frame[list(detect[index][u'faceRectangle'].values())[0]: (list(detect[index][u'faceRectangle'].values())[0] + list(detect[index][u'faceRectangle'].values())[
                                        3]), list(detect[index][u'faceRectangle'].values())[1]:(list(detect[index][u'faceRectangle'].values())[1] + list(detect[index][u'faceRectangle'].values())[2])]
                    name_crop=str(today)+"1-"+current_time+str(randint(0, 100))+"-crop.jpg"
                    cv2.imwrite("data/"+name_crop, crop_img)
                    storecrop(name_crop)
                    person=requests.get(uriPerson,  headers = header)
                    nameperson=person.json()[u'name']

                    mongo(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceAttributes'], detect[index][u'faceRectangle'], (
                        "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                    os.remove("data/"+name_crop)
                    
            os.remove("data/"+name)
        
now=datetime.now()

current_time=now.strftime("%H%M%S")


# #print(int(t2(12, 30).strftime("%H%M")) > int(datetime.now().strftime("%H%M")))
count1=1
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
while(True):
    # ##print("a")
    # try:
    ret, img=cap.read()
    # print("count",count1)
    # print("time",datetime.now().strftime("%H:%M:%S"))
    # if(img  is not None):
        # if ((cv2.waitKey(20) & 0xFF == ord('q')) | (int(t2(20,00).strftime("%H%M"))<int(datetime.now().strftime("%H%M")))):
    if (cv2.waitKey(20) & 0xFF == ord('q')):
            break
        # if (cv2.waitKey(20) & 0xFF == ord('q')) | (not ret):
        #     break
        # asyncio.run(imagescan(img, count1))
        # executor.submit(asyncio.run(imagescan(img, count1)))
        # executor.submit(imagescan(img, count1))
    _thread.start_new_thread(imagescan, (img, count1))
    count1=count1 + 1
    # else:
    #     count1=count1 + 1
	# except Exception as ex:
    #     count1=count1 + 1
    #     pass

cap.release()
cv2.destroyAllWindows()

# sys.stdout = sys.__stdout__
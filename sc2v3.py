import sys 
from datetime import  timedelta, datetime, date, time as t2
import time

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
import _thread

import requests
import json
# from datetime import  timedelta, datetime, date, time as t2
# import time
from random import randint
# set to your own subscription key value
subscription_key = '99d0310d30c24046a148cbf795a34121'

blob_service_client = BlobServiceClient.from_connection_string(
    "DefaultEndpointsProtocol=https;AccountName=oneteamblob;AccountKey=qcv7bSwg5vFNZRt1gY9XLPcv6OWKdKakKCj5znpUQRNQTPAOkLbhnCuZpt/1m4Gc9f5tV55x0CEzcVWjCubTaQ==;EndpointSuffix=core.windows.net")
# cap = cv2.VideoCapture("rtsp://admin:admin@10.76.53.14:8554/stream0/out.h264")

# Create a unique name for the container
container_name = "facedetection"

net = cv2.dnn.readNetFromDarknet('./yolov3-face.cfg', './yolov3-wider_16000.weights')
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# def storeblob(name):
#     print(name)
#     blob_client = blob_service_client.get_blob_client(
#                 container=container_name, blob=name)
#     with open("data/"+name, "rb") as data:
#         blob_client.upload_blob(data) 

def encrypt_val(clear_text):
    master_key = b'meafacialkeycam1' 
    encoder = PKCS7Encoder()
    raw = encoder.encode(clear_text)
    iv = Random.new().read( 16 )
    cipher = AES.new( master_key, AES.MODE_CBC, iv, segment_size=128 )
    return base64.b64encode( iv + cipher.encrypt( raw ) ) 

def storecrop(name,now):
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
        "data": img_enc_str,
        "date": now.strftime("%Y-%m-%d")
    }
    )


    

def infocrop(name,now,detectname,confidence):
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    db2 = client.cropinfo
    db2.data.insert_one({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "name": name,
        "datetime": now.strftime("%Y%m%d%H%M%S"),
        "detected": detectname,
        "confidence": confidence,
        "train": '',
        "camera": 2
    }
    )



def apiidentify(name, arrfaceid):

    identify_url = 'https://meafacedetection.cognitiveservices.azure.com/face/v1.0/identify'
    header = {'Ocp-Apim-Subscription-Key': subscription_key}
    paramsIden = json.loads('{ "personGroupId": "mea", "faceIds": '+json.dumps(
        arrfaceid)+',"confidenceThreshold": 0.1, "maxNumOfCandidatesReturned": 1 }')
    return requests.post(identify_url,  headers=header, json=paramsIden)

def apidetect(img):
    face_api_url = 'https://meafacedetection.cognitiveservices.azure.com/face/v1.0/detect'

#     image_url = "https://oneteamblob.blob.core.windows.net/facedetection/"+name
    headers = {'Content-Type': 'application/octet-stream','Ocp-Apim-Subscription-Key': subscription_key}

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'true',
        'returnFaceAttributes': 'emotion,gender,age,blur',
        
    }
    return requests.post(face_api_url, params=params, headers=headers, data=img)


def apidetect2(img):
    face_api_url = 'https://meafacedetection.cognitiveservices.azure.com/face/v1.0/detect'

#     image_url = "https://oneteamblob.blob.core.windows.net/facedetection/"+name
    headers = {'Content-Type': 'application/octet-stream','Ocp-Apim-Subscription-Key': subscription_key}

    params = {
        'returnFaceId': 'true',
        # 'returnFaceLandmarks': 'true',
        # 'returnFaceAttributes': 'emotion,gender,age,blur',
        'detectionModel': 'detection_02',
    }
    return requests.post(face_api_url, params=params, headers=headers, data=img)
    

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
    today = now.strftime("%Y-%m-%d")
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    db = client.checkin
    emo = getemo(faceAttributes['emotion'])
    query = {"id": nameperson}
    queryy = db.checkin[today].find(query)
    


    if(queryy.count() > 0):
        db_default = client.mea
        query_default = {"id": nameperson}
        default_data = db_default.default.find_one(query_default)
        if(default_data['gender'] == faceAttributes['gender']):
            newvalues = { "$set": { "cameraout": 2,"checkout": checkin ,"checkoutEmotion": { "gender": default_data['gender'], "age": faceAttributes['age']+ int(default_data['margin']), "emotion": faceAttributes['emotion'] },"checkoutEmo": emo, "checkoutImageCrop": imageCropUrl , "checkoutdatetime": now.strftime("%Y%m%d%H%M%S")} }
            db.checkin[today].update_one(query, newvalues)

            db.checkattendance.update_one(query, { "$set": {"checkout": { "time": now.strftime("%H:%M:%S"),
                        "emotion": { "gender": default_data['gender'], "age": faceAttributes['age']+ int(default_data['margin']), "emotion": faceAttributes['emotion'] }
            }} }) 

def mongo2(now,timei, nameperson, checkin, faceRectangle, image_url, imageCropUrl):
    today = now.strftime("%Y-%m-%d")
    year_today = int(now.strftime("%Y"))
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    db = client.checkin
    query = {"id": nameperson}
    queryy = db.checkin[today].find(query)

    


    if(queryy.count() > 0):
        db_default = client.mea
        query_default = {"id": nameperson}
        default_data = db_default.default.find_one(query_default)
        
        newvalues = { "$set": { "cameraout": 2,"checkout": checkin ,"checkoutEmotion": { "gender": default_data['gender'], "age": (year_today - 1958 - int(default_data['year'])) + int(default_data['margin']), "emotion": { "anger": 0, "contempt": 0, "disgust": 0, "fear": 0, "happiness": 0, "neutral": 1, "sadness": 0, "surprise": 0 } },"checkoutEmo": "neutral", "checkoutImageCrop": imageCropUrl , "checkoutdatetime": now.strftime("%Y%m%d%H%M%S")} }
        db.checkin[today].update_one(query, newvalues)

        db.checkattendance.update_one(query, { "$set": {"checkout": { "time": now.strftime("%H:%M:%S"),
                    "emotion": "neutral"
        }} }) 


def mongodetect(now,timei, nameperson, checkin, faceAttributes, faceRectangle, image_url, imageCropUrl):
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    today = now.strftime("%Y-%m-%d")
    emo = getemo(faceAttributes['emotion'])
    query = {"id": nameperson,"cameraout":2}
    db2 = client.detect
    db_default = client.mea
    query_default = {"id": nameperson}
    default_data = db_default.default.find_one(query_default)
    if(default_data['gender'] == faceAttributes['gender']):
        db2.detect[today].update(
            query,
            {
                "id": nameperson,
                "checkin": "",
                "checkindatetime": "",
                "checkinEmotion": {"gender":"","age":0},
                "checkinEmo": emo,
                "checkinImageCrop": "",
                "camerain": 0,
                "checkout": checkin,
                "checkoutEmotion": { "gender": default_data['gender'], "age": faceAttributes['age']+ int(default_data['margin']), "emotion": faceAttributes['emotion'] },
                "checkoutEmo": emo,
                "checkoutImageCrop": imageCropUrl,
                "cameraout": 2,
                "checkoutdatetime": now.strftime("%Y%m%d%H%M%S"),
            },
                upsert=True
            )   
    
def mongodetect2(now,timei, nameperson, checkin, faceRectangle, image_url, imageCropUrl):
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    today = now.strftime("%Y-%m-%d")
    year_today = int(now.strftime("%Y"))
    query = {"id": nameperson,"cameraout":2}
    db2 = client.detect
    db_default = client.mea
    query_default = {"id": nameperson}
    default_data = db_default.default.find_one(query_default)
    db2.detect[today].update(
    query,
    {
        "id": nameperson,
        "checkin": "",
        "checkindatetime": "",
        "checkinEmotion": {"gender":"","age":0},
        "checkinEmo": "",
        "checkinImageCrop": "",
        "camerain": 0,
        "checkout": checkin,
        "checkoutEmotion": { "gender": default_data['gender'], "age": (year_today - 1958 - int(default_data['year'])) + int(default_data['margin']), "emotion": { "anger": 0, "contempt": 0, "disgust": 0, "fear": 0, "happiness": 0, "neutral": 1, "sadness": 0, "surprise": 0 } },
        "checkoutEmo": "neutral",
        "checkoutImageCrop": imageCropUrl,
        "cameraout": 2,
        "checkoutdatetime": now.strftime("%Y%m%d%H%M%S"),
    },
        upsert=True
    )

def getprofile(faceid):
    client = pymongo.MongoClient(
            "mongodb://127.0.0.1:27017")
    db_profile = client.mea
    query_faceid = {"faceid": faceid}
    profile_data = db_profile.profile.find_one(query_faceid, {'_id':0,'encimage': 0})
    return profile_data

    
    


def imagescan(img, count,now):
    if (count % 56) == 0:
        print("count",count)
        current_time=now.strftime("%H%M%S")
        name=now.strftime("%Y-%m-%d")+"-2-"+current_time+str(count%60)+".jpg"
        cv2.imwrite("data/"+name, img)
        frame = cv2.imread("data/"+name)
        # framesize = os.path.getsize("data/"+name)
        # if(framesize > 200000):
        #     print("not gray",count)
        # else:
        #     storeblob(name)
        #     requests.get('http://localhost:3000/frameerror/2')
        #     os.remove("data/"+name)
        #     return False
        find = False
        # (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        countperson = 0
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if((classID == 0) & (confidence>0.5)):
                    find = True
                    countperson = countperson + 1
                    # print(confidence)
        print("countperson",countperson)
        if(find & (countperson <6)):
            # storeblob(name)
            bitimage = open("data/"+name, 'rb').read()
            response=apidetect(bitimage)
            detect=response.json()
            print("detect1",detect)
            if(detect != []):
                arrfaceid=[]
                for face in detect:
                    arrfaceid.append(face[u'faceId'])
                response=apiidentify(name, arrfaceid)
                identify=response.json()
                for index, iden in enumerate(identify):
                    uriPerson='https://meafacedetection.cognitiveservices.azure.com/face/v1.0/persongroups/mea/persons/' + \
                        str(json.dumps(identify[index][u'candidates'][0][u'personId'])).replace(
                            '"', '')
                    header={'Ocp-Apim-Subscription-Key': subscription_key}
                    crop_img=frame[list(detect[index][u'faceRectangle'].values())[0]: (list(detect[index][u'faceRectangle'].values())[0] + list(detect[index][u'faceRectangle'].values())[
                                        3]), list(detect[index][u'faceRectangle'].values())[1]:(list(detect[index][u'faceRectangle'].values())[1] + list(detect[index][u'faceRectangle'].values())[2])]
                    name_crop=now.strftime("%Y-%m-%d")+"-2-"+current_time+str(count%60)+str(index)+"-crop.jpg"
                    cv2.imwrite("data/"+name_crop, crop_img)
                    storecrop(name_crop,now)
                    prof = getprofile(identify[index][u'candidates'][0][u'personId'])
                    conf = prof['individual_confidence']
                    if(identify[index][u'candidates'][0][u'confidence'] > float(conf)):
                        nameperson=prof['id']
                        # person=requests.get(uriPerson,  headers = header)
                        # nameperson=person.json()[u'name']
                        mongodetect(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceAttributes'], detect[index][u'faceRectangle'], (
                            "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                        
                        mongo(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceAttributes'], detect[index][u'faceRectangle'], (
                            "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                        # mongo(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceRectangle'], (
                    #      "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                        infocrop(name_crop,now,nameperson,identify[index][u'candidates'][0][u'confidence']) 
                        requests.get('http://localhost:3000/walkoutalertbyid/'+nameperson)
                        # requests.get('http://localhost:3000/walkinalertbyid/'+nameperson)
                    else:
                        # person=requests.get(uriPerson,  headers = header)
                        # nameperson=person.json()[u'name']
                        nameperson=prof['id']
                        infocrop(name_crop,now,nameperson,identify[index][u'candidates'][0][u'confidence']) 
                    os.remove("data/"+name_crop)
            else:
                response=apidetect2(bitimage)
                detect=response.json()
                print("detect2",detect)
                if(detect != []):
                    arrfaceid=[]
                    for face in detect:
                        arrfaceid.append(face[u'faceId'])
                    response=apiidentify(name, arrfaceid)
                    identify=response.json()
                    for index, iden in enumerate(identify):
                        uriPerson='https://meafacedetection.cognitiveservices.azure.com/face/v1.0/persongroups/mea/persons/' + \
                            str(json.dumps(identify[index][u'candidates'][0][u'personId'])).replace(
                                '"', '')
                        header={'Ocp-Apim-Subscription-Key': subscription_key}
                        crop_img=frame[list(detect[index][u'faceRectangle'].values())[0]: (list(detect[index][u'faceRectangle'].values())[0] + list(detect[index][u'faceRectangle'].values())[
                                            3]), list(detect[index][u'faceRectangle'].values())[1]:(list(detect[index][u'faceRectangle'].values())[1] + list(detect[index][u'faceRectangle'].values())[2])]
                        name_crop=now.strftime("%Y-%m-%d")+"-2-"+current_time+str(count%60)+str(index)+"-crop.jpg"
                        cv2.imwrite("data/"+name_crop, crop_img)
                        storecrop(name_crop,now)
                        prof = getprofile(identify[index][u'candidates'][0][u'personId'])
                        conf = prof['individual_confidence']
                        if(identify[index][u'candidates'][0][u'confidence'] > float(conf)):
                            # person=requests.get(uriPerson,  headers = header)
                            # nameperson=person.json()[u'name']
                            nameperson=prof['id']
                            mongodetect2(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceRectangle'], (
                                "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                            # mongo(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceAttributes'], detect[index][u'faceRectangle'], (
                            #     "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                            mongo2(now,now.strftime("%H:%M"), nameperson, now.strftime("%H:%M"), detect[index][u'faceRectangle'], (
                                "https://oneteamblob.blob.core.windows.net/facedetection/"+name), name_crop)
                            infocrop(name_crop,now,nameperson,identify[index][u'candidates'][0][u'confidence']) 
                            requests.get('http://localhost:3000/walkoutalertbyid/'+nameperson)
                            # requests.get('http://localhost:3000/walkinalertbyid/'+nameperson)
                        else:
                            # infocrop(name_crop,now,"",0)
                            # person=requests.get(uriPerson,  headers = header)
                            # nameperson=person.json()[u'name']
                            nameperson=prof['id']
                            infocrop(name_crop,now,nameperson,identify[index][u'candidates'][0][u'confidence'])  
                        os.remove("data/"+name_crop)

            # os.remove("data/"+name)
            # print("done1")
        os.remove("data/"+name)
        print("done",count)
    


      

count1=1
print("new running")
while(True):
    print("connect camera...")
    cap = cv2.VideoCapture("rtsp://admin:admin@10.76.53.15:8554/stream0/out.h264")
    while(True):
        ret, img=cap.read()
        timenow =datetime.now() + timedelta(hours=7)
        bool1 = ((int(t2(5,45).strftime("%H%M"))<int( (timenow).strftime("%H%M")) ) & (int(t2(20,00).strftime("%H%M"))>int( (timenow).strftime("%H%M")) )  ) & ((timenow).weekday() < 5)
        # bool1 = True
        # bool1 = False
        if ((cv2.waitKey(20) & 0xFF == ord('q')) | (not bool1)):
        # if ((cv2.waitKey(20) & 0xFF == ord('q'))):
            break
        else:
            if ret:
                _thread.start_new_thread(imagescan, (img, count1,timenow))
                # imagescan(img, count1,timenow)
            else:
                client = pymongo.MongoClient(
                    "mongodb://127.0.0.1:27017")
                db2 = client.errorlog
                errdate = (datetime.now() + timedelta(hours=7))
                db2.python[errdate.strftime("%Y-%m-%d")].insert_one({
                    "datetime": errdate.strftime("%Y%m%d%H%M%S"),
                    "message": "Camera 2 not avaliable"
                }
                )
                break
        count1=count1 + 1

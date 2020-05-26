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

timenow =datetime.now() + timedelta(hours=7)
print(timenow)
print(( (timenow).strftime("%H%M")) )
print(((int(t2(5,00).strftime("%H%M")))))
print(int(t2(20,00).strftime("%H%M")))
print(((int(t2(5,00).strftime("%H%M"))<int( (timenow).strftime("%H%M")) )))
print((int(t2(20,00).strftime("%H%M"))>int( (timenow).strftime("%H%M")) ))
print("old",((int(t2(5,00).strftime("%H%M"))>int( (timenow).strftime("%H%M")) ) | (int(t2(20,00).strftime("%H%M"))<int( (timenow).strftime("%H%M")) )  ))
print("new",((int(t2(5,00).strftime("%H%M"))<int( (timenow).strftime("%H%M")) ) & (int(t2(20,00).strftime("%H%M"))>int( (timenow).strftime("%H%M")) )  ))
print("full",((int(t2(5,00).strftime("%H%M"))<int( (timenow).strftime("%H%M")) ) & (int(t2(20,00).strftime("%H%M"))>int( (timenow).strftime("%H%M")) )  ) & (timenow).weekday() < 5)


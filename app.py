
from flask import Flask, jsonify,render_template,request,Response,redirect,flash
import threading
import os
# from werkzeug.utils import secure_filename
import serial as sr
# from mailsend import sendmail


import cv2
import numpy as np
from PIL import Image
#import systemcheck

# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model


IMAGE_SIZE = (224, 224, 3)
CATEGORIES = {0:'Snake detected',1:'No snake detected'}

model = load_model("Snake_detection_resnet_50_model.h5")

########################################################################################
def model_warmup():
    dummy_image = []
    for i in range(224):
        dummy_image.append([[0]*3]*224)
    image = np.array(dummy_image)
    # print(image.shape)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    # print(pred)
########################################################################################
def predict_snake(img):
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    img_rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_flip_ver = cv2.flip(img, 0)
    img_flip_hor = cv2.flip(img, 1)
    images = []
    images.append(img)
    images.append(img_rotated_90)
    images.append(img_rotated_180)
    images.append(img_rotated_270)
    images.append(img_flip_ver)
    images.append(img_flip_hor)
    images = np.array(images)
    images = images.astype(np.float32)
    images /= 255
    op = []
    # make predictions on the input image
    for im in images:
        image = np.array(im)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
        op.append(pred)
        # print("Pred:", pred, CATEGORIES[pred])
    op = np.array(op)
    print("Final Output:", CATEGORIES[np.bincount(op).argmax()])
    
    # if CATEGORIES[np.bincount(op).argmax()] == 0:
    #     sendmail("Snake in your farm")
    

    return CATEGORIES[np.bincount(op).argmax()]   

model_warmup()

########################################################################################

DB = {
    "msg":"",
    "flag" : 0
}

########################################################################################
def checkPort():
    for i in range(0,150):
        try:
            ser = sr.Serial(f'COM{i}',9600,timeout=1)
            print(f'COM{i}')
            return ser
        except Exception as e:
            pass
########################################################################################

def getData():
    global DB
    port = checkPort()
    while 1:
        data = port.readline().decode("utf-8")
        # print(data)
        if len(data) > 0:
            try:
                data = data.split(",")
                # print(data)
                if int(data[1]) == 0 and int(data[2]) == 1 and DB["flag"] == 0:
                    DB["flag"] = 1
                # update(d)
            except Exception as e:
                print(e)

########################################################################################

app = Flask(__name__)
video = cv2.VideoCapture(0)## 0 for internal  : 1 for external


@app.route('/',methods=["get","post"])
def server_app():
    global DB
    if request.method=="POST":
        return DB
    return render_template("index.html")
########################################################################################
@app.route('/clear',methods=["get","post"])
def Clear():
    global DB
    DB["flag"] = 0
    DB["msg"] = ""
    return "OK"
########################################################################################
def gen(video):
    global DB
    cam=cv2.VideoCapture(0)## 0 for internal  : 1 for external
    cam.set(3,1080)
    cam.set(4,1080)
    while True:
        try:
            _,image=cam.read()
            image = cv2.flip(image, 1)
            # DB["msg"] = predict_snake(image)
            if DB["flag"] == 1:
                DB["flag"] = 2
                DB["msg"] = predict_snake(image)
                cv2.imwrite("img.png",image)
                if(CATEGORIES[0] == DB["msg"]):
                    os.system("python mailsend.py")
            # print()
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        except Exception as e:
            print("Error : ",e)
########################################################################################
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

kwargs = {'host': '0.0.0.0', 'port': 8080, 'threaded': True, 'use_reloader': False, 'debug': True}
if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs=kwargs).start()
    threading.Thread(target=getData).start()
    #app.run(port=6050)
        
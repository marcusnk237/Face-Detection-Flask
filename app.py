################# Loading libraries and frameworks #################
from flask import Flask, render_template, Response
import cv2
import os
import json
#####################################################################

app = Flask(__name__)

################################## Haar Detector path ##################################
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
#########################################################################################

############### Haar Classifier creation ###############
face_class = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
########################################################

n_faces = 0

#################### Get video ####################
## Video path. By default, we get the webcam ## 
video_path=0
###################################################
camera = cv2.VideoCapture(video_path)
###################################################

# Function which allows us to detect the face #
def detect_face():  
    while True:
        ######### Get every frame#######
        success, frame = camera.read()
        ################################
        # Stop if an error occurs #
        if not success:
            break
        #########################
        ######## Conversion into grey level ########
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ##############################################
        ###################################### Face detection ###################################
        faces = face_class.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        #########################################################################################
        
        ####### Boundaries boxes creation for every detected face #######
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            roi_color = frame[fy:fy+fh, fx:fx+fw]
        ####################################################################
        ################# Convert the result into an image the browser can ddisplay #################
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        #######################################################################################
############################################

######################## Routing to the face detection function ########################
@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################################################

################# Main page #################
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')
#############################################
if __name__ == '__main__':
    app.run(debug=False)

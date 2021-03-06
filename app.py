################# Importation des bibliothèques Flask et Python #################
from flask import Flask, render_template, Response
import cv2
import os
import json
#################################################################################

app = Flask(__name__)

############################## Chemin du détecteur de Haar ##############################
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
#########################################################################################

############### Créateur du classifieur de Haar ###############
face_class = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
###############################################################

n_faces = 0

#################### Récupération du flux vidéo ##################
## Chemin du flux vidéo. Par défaut nous avons choisi la webcam ## 
video_path=0
##################################################################
camera = cv2.VideoCapture(video_path)
##################################################################

# Fonction assurant la détection de visage #
def detect_face():  
    while True:
        # Récupération frame par frame #
        success, frame = camera.read()
        ################################
        # Arrêt en cas d'erreur #
        if not success:
            break
        #########################
        ######## Conversion en niveau de gris ########
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ##############################################
        ################################## Détection des visages ################################
        faces = face_class.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        #########################################################################################
        
        ####### Création de Boundaries Box pour les visages détectés #######
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            roi_color = frame[fy:fy+fh, fx:fx+fw]
        ####################################################################
        ################# Conversion du resultat en image lisible par le navigateur #################
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        #############################################################################################
        ############ Concatenation des images pour former une seconde d'image #################
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        #######################################################################################
############################################

#################### Routage vers la fonction de détection de visage ####################
@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################################################

################# Page d'accueil #################
@app.route('/')
def index():
    """Page d'accueil."""
    return render_template('index.html')
###################################################

if __name__ == '__main__':
    app.run(debug=True)

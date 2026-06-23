#------------------------------------Libraries------------------------------------#

#Computer vision library for camera access, frame reading and opening window
import cv2

#Library used for face and landmark detection
import mediapipe as mp

#This library is used for distance measure (ear ratio measure) and mathematical calculation
import numpy as np

#It will calculate time
import time

#It will play alarm sound when driver sleeps
import pygame

#This will play alarm sound in background so real time image will not freeze
import threading

#checks whether file for aarm sound exists or not
import os


#------------------------------------Load Sound------------------------------------#

#sound system starts means now audio part is activated and sound can be load and played
pygame.mixer.init()


#----------------------------------EAR Calculation---------------------------------#

#Function that will calculate the ear
def calculate_ear(eye_landmarks):

    #calculate vertical distance V1 between point 1 and 5
    V1=np.linalg.norm(eye_landmarks[1]-eye_landmarks[5])

    #calculate vertical distance V2 between point 2 and 4
    V2=np.linalg.norm(eye_landmarks[2]-eye_landmarks[4])

    #calculate the horizontal distance H 
    H=np.linalg.norm(eye_landmarks[0]-eye_landmarks[3])

    #Calculate eye aspect ration (EAR)
    return(V1+V2)/(H*2.0)


#------------------------------------Alarm Sound-----------------------------------#

#Function that will play the alrm sound
def alarm_sound(path):

    #Condition: if file exists
    if os.path.exists(path):
        pygame.mixer.music.load(path)   #sound file is loaded
        pygame.mixer.music.play()       #play the alarm sound
        pygame.mixer.music.play(-1)     #Infinite loop


#------------------------------------Alarm Sound-----------------------------------#

#Access the face mesh model inside mediapipe library
mp_face_mesh=mp.solutions.face_mesh

#model is activated and and AI model is ready to detect live face
face_mesh=mp_face_mesh.FaceMesh(
    max_num_faces=1,        #detect maximum(only) one face at a time
    refine_landmarks=True   #detect much accurate landmarks
)


#------------------------------------Landmarks------------------------------------#

#Defined fix landmark index numbers

#Lanmarks point numbers of left eye
LEFT_EYE=[33,160,158,133,153,144]

#Lanmarks point numbers of right eye
RIGHT_EYE=[362,385,387,263,373,380]


#-----------------------------------Calibration-----------------------------------#

#Calibration time is 5 seconds, eyes mus be opened so that AI model calculates the EAR ratio
CALIBRATION_TIME=5

#If eyes are closed upto 5 seconds, alarm will be played
ALARM_THRESHOLD_TIME=5

#EAR values would be stored in that list/array
ear_baseline_list=[]


#-----------------------------------Camera Start-----------------------------------#

#Camera start--> 1: beacause external camera is attached, 0: is PC's camera is used
cap=cv2.VideoCapture(0)

#if camera does not open
if not cap.isOpened():
    print("Camera not detected!")
    exit()


#----------------------------Time Start for Calibration----------------------------#

#Time started for calibration
start_time=time.time()


calibration_done=False  #Will be true after 5 seconds are done
alarm_on=False          #Alarm will start (true) when strong drowsiness is detected
eye_closed_start=None   #Record the time for the closure of eye
ear_baseline=0          #Store average value of normal open-eye EAR
ear_history=[]          #List that will store ear history
sensitivity_window=50   #Record ear for 50 frames
sensitivity=0.70        #Default sesitivity 0.70

#-----------------------------------Infinite Loop----------------------------------#

while True:
    ret, frame=cap.read()   #Stores image (ret= true/false, frame= actual image)
    
    #If no image is detected then program/loop will be stopped
    if not ret:
        break

    frame_h, frame_w=frame.shape[:2]    #Calculates height and width of image

    #Converts BGR to RGB (Mediapipe uses RGB where as OpenCV uses BGR)
    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Detects facial landmarks
    results=face_mesh.process(frame_rgb)

    status_text=""                  #This variables stores the message showed on screen
    background_color=(255,0,0)      #Default background color is blue (calibration mode)


#----------------------------------Face Detection----------------------------------#

    #Condition is true when face is detected
    if results.multi_face_landmarks:
        
        #Collecting all landmark points of detected face (0-->468)
        mesh_points=results.multi_face_landmarks[0].landmark

        #Findling left eye pixel coordinates (0 to 6)
        left_eye=np.array([[mesh_points[p].x*frame_w,mesh_points[p].y*frame_h] for p in LEFT_EYE])

        #Finding pixel coordinates for right eye (0 to 6)
        right_eye=np.array([[mesh_points[p].x*frame_w,mesh_points[p].y*frame_h] for p in RIGHT_EYE])

        #Calculating EAR 
        ear=(calculate_ear(left_eye)+calculate_ear(right_eye))/2.0


#------------------------------------Calibration-----------------------------------#

#Calibration not done means system is still recording baseline
        if not calibration_done:
            ear_baseline_list.append(ear)               #Storing each frame of ear
            status_text="Calibration: Keep Eyes Open"   #Message will be displayed on screen
            if time.time()-start_time>CALIBRATION_TIME: #Checks if 5 seconds are done for calibration
                ear_baseline=np.mean(ear_baseline_list) #Taking average of EAR values collected in previous 5 seconds
                calibration_done=True                   #Calibration completed, now system will switch to monitoring mode


#------------------------------------Monitoring------------------------------------#
        
        #Else true means now calibration is completed and system is in the monitoring mode
        else:
            ear_history.append(ear)                 #Stores recent EAR values for adaptive sensitivity calculation
            if len(ear_history)>sensitivity_window: #Only 50 frames would be stored to maintain a fixed window size
                ear_history.pop(0)                  #Remove oldest value so only recent 50 frames are stored

            if len(ear_history)==sensitivity_window:#Checks if enough data is collection to calculate variation
                ear_std=np.std(ear_history)         #Calculate standard deviation (variation in eye movement)

                #Adjust sensitivity based on variation (Adaptive sensitivity)
                if ear_std>0.02:        #If variation is greater than sensitivity (0.02)
                    sensitivity=0.75    #Means more variation -> increase the sensitivity
                elif ear_std<0.01:      #If variation is less than the sensitivity (0.01)
                    sensitivity=0.65    #Means less variation -> sensitivity would be decreased
                else:                   
                    sensitivity=0.70    #normal condition, means default sensitivity

            #Dynamic threshold based on baseline EAR and adaptive sensitivity
            threshold=ear_baseline*sensitivity      #Threshold = normal eye open * sensitivity
            if ear<threshold:                       #If current EAr less than threshold means eyes are closed or half closed
            

                
#---------------------------------Eye Closure Time---------------------------------#

                if eye_closed_start is None:        #If true means detected that eye are closed
                    eye_closed_start=time.time()    #Soring currect time of closed eyes
                elapsed=time.time()-eye_closed_start#Calculate how long eyes have been continuously closed


#----------------------------------Mild Drowsiness---------------------------------#

                if elapsed<ALARM_THRESHOLD_TIME:    #If closed eyes time < 5 seconds (short drowsiness/blink)
                    status_text="Mild Drowsiness"   #Message displayed on screen
                    background_color=(0,255,255)    #Background color changed to yellow


#--------------------------------Strong Drowsiness---------------------------------#          
    
                #Else eyes are closed more than 5 seconds= serious sleep condition
                else:
                    status_text="SLEEP ALERT! WAKE UP!" #Message displayed on screen
                    background_color=(0,0,255)          #Backgroung color becomes red
                    if not alarm_on:                    #Checks if alarm is not working already
                        alarm_on=True                   #Now alarm is working


#------------------------------Background Alarm Thread-----------------------------#          

                        #Start alarm in seperate thread (background) so video stream does not freeze
                        threading.Thread(               
                            target=alarm_sound,         #Function of alarm is working
                            args=("alarm.mp3",),        #Providing argument to function
                            daemon=True                 #If main background is stopped then thred will also be closed automatically
                        ).start()                       #Thread started


#--------------------------------Eyes Open Condition-------------------------------# 

            else:   #Means eyes are open again (ear>=threshold)
                eye_closed_start=None           #Eye closure time reset
                alarm_on=False                  #Alarm flag reset
                pygame.mixer.music.stop()       #Stop alarm immediately when eyes open
                status_text="Driver Normal"     #Message displayed on screen
                background_color=(0,255,0)      #Background colour green


#---------------------------------Overlay Creation---------------------------------#

    #Generating a copy of original camera fram o prevent direct modification of original image
    overlay=frame.copy()
    
    #Solid colour layer is drawn on whole screen
    cv2.rectangle(overlay, (0,0), (frame_w, frame_h), background_color, -1)

    #Transparent coloured effect (alert look)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


#-----------------------------------Display Text-----------------------------------#
   
    #Showing status text on screen
    cv2.putText(frame, status_text, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    #Displaying final output window
    cv2.imshow("AI Drowsiness Monitoring System", frame)

    #Loop will break if ESC key (27) is pressed
    if cv2.waitKey(1) & 0xFF==27:
        break


#--------------------------------Release Resources---------------------------------#

cap.release()               #Closing Camera
cv2.destroyAllWindows()     #Closing all OpenCV windows
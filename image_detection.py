import cv2
import mediapipe as mp
import tensorflow as tf

mp_face_detection = mp.solutions.face_detection

model = tf.keras.models.load_model('model1.h5')

emotion_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'suprised']

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  
    image = cv2.imread('test_image.jpg') 
    image = cv2.resize(image, (480, 640))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        x_start, y_start = int(box.xmin * image.shape[1]) - 5 , int(box.ymin * image.shape[0]) - 50
        x_end, y_end = int((box.xmin + box.width) * image.shape[1])+5 , int((box.ymin + box.height)* image.shape[0])
        img_new = image[y_start:y_end, x_start:x_end]
        
        try : 
          img_new = cv2.resize(img_new, (48, 48))
          img_new = img_new.reshape((1,) + img_new.shape)
          img_new = img_new/255.0 
          predictions = model.predict(img_new) 
          id = predictions.argmax(axis=1)[0]
          print(emotion_names[id])
          cv2.putText(image, emotion_names[id],(x_start, y_start - 10), color=(255,0,0), fontScale=2, thickness=2, fontFace=cv2.FONT_HERSHEY_DUPLEX)
          cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,255), 5)
          
        except Exception as e : 
          print(e) 
        
        # score = round(detection.score[0]*100)
        # cv2.putText(image, str(score) + "%",(x_start, y_start), color=(255,0,0), fontScale=2, thickness=2, fontFace=cv2.FONT_HERSHEY_DUPLEX)
        # cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,255), 5)
        # mp_drawing.draw_detection(image, detection)
    cv2.imshow('Emotion Detection', image)
    cv2.imwrite('new_image.png', image) 
    if cv2.waitKey(0) & 0xFF == ord('q') : 
        exit() 

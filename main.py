from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os
import cv2
import pandas as pd
from utils import *

#Download Model Weights
!curl -L "https://drive.usercontent.google.com/download?id={1--ZhI43tn9NdB-dDmYxP0xDFP873GX07}&confirm=xxx" -o "trained_resnet_state_dict.pth"
!curl -L "https://drive.usercontent.google.com/download?id={1FL9kSke0UWUC4VDKFez5Uv9MqHqQj3RL}&confirm=xxx" -o "yolov8m.pt"

data_transform = A.Compose([
    NumpyToTensor()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

yolo_model = YOLO('yolov8m.yaml').load('yolov8m.pt')

class_map = {'Komolafe': 0, 'Mac': 1, 'Adedayo': 2,
             'Yisau': 3,'Cynthia': 4, 'Joseph': 5,
             'Michael': 6,'Jason': 7,'Ogechi': 8, 
             'Emmanuel': 9,'Mubaraq': 10,'Nasir': 11}

resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(class_map)
).to(device)

resnet.load_state_dict(torch.load('trained_resnet_state_dict.pth'))

resnet.eval()

if os.path.exists('Attendance_Records.csv'):
    df = pd.read_csv('Attendance_Records.csv')
else:
    df = pd.DataFrame(columns=['Facilitator', 'Starting time', 'Closing time'])

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(source=frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.xyxy[0]
        boxes = boxes.cpu().numpy()
        x1, y1, x2, y2 = boxes
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped_image = frame[y1:y2, x1:x2]
        cropped_image = cv2.resize(cropped_image, (224, 224))
        t_image = data_transform(image=cropped_image)['image']
        t_image = t_image.unsqueeze(0)

        outputs = resnet(t_image.to(device))
        _, preds = torch.max(outputs, 1)
        pred_class = get_key(preds.item(), class_map)

        # Check if the class has been identified before
        split = []
        if pred_class in df['Facilitator'].values:
            df.loc[df['Facilitator'] == pred_class, 'Closing time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            split.append({'Facilitator': pred_class, 'Starting time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                          'Closing time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            temp_df  = pd.DataFrame(split, columns=['Facilitator', 'Starting time', 'Closing time'])
            df = pd.concat([temp_df, df])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{pred_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Real-Time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df.to_csv('Attendance_Records.csv', index=False)

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

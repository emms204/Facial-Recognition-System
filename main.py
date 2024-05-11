from facenet_pytorch import MTCNN, InceptionResnetV1
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
# !curl -L "https://drive.usercontent.google.com/download?id={1--ZhI43tn9NdB-dDmYxP0xDFP873GX07}&confirm=xxx" -o "trained_resnet_state_dict.pth"
# !curl -L "https://drive.usercontent.google.com/download?id={1FL9kSke0UWUC4VDKFez5Uv9MqHqQj3RL}&confirm=xxx" -o "yolov8m.pt"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SiameseNetwork().to(device)

model.load_state_dict(torch.load('InceptionResnetV1_state_dict.pth', map_location=device))
mtcnn = MTCNN(keep_all=True, device=device)

model.eval()

if os.path.exists('Attendance_Records.csv'):
    df = pd.read_csv('Attendance_Records.csv')
else:
    df = pd.DataFrame(columns=['Facilitator', 'Starting time', 'Closing time'])

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = mtcnn.detect(frame)

        # Process each face individually
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = frame[y1:y2, x1:x2]
                cropped_image = cv2.resize(cropped_image, (224, 224))
                t_image = val_transform(image=cropped_image)['image']
                t_image = t_image / 255.0
                t_image = t_image.unsqueeze(0)

                # Get results for the current face
                results = loop_classes('Cropped_Images', model, t_image)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Check if results are available before accessing
                if results:
                    label_text = f'{results[0]}'
                else:
                    label_text = 'Unknown'

                # Check if the class has been identified before
                split = []
                if label_text in df['Facilitator'].values:
                    df.loc[df['Facilitator'] == label_text, 'Closing time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    split.append({'Facilitator': label_text, 'Starting time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'Closing time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                    temp_df  = pd.DataFrame(split, columns=['Facilitator', 'Starting time', 'Closing time'])
                    df = pd.concat([temp_df, df])

                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLO Real-Time Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    df.to_csv('Attendance_Records.csv', index=False)
    cap.release()
    cv2.destroyAllWindows()

        



from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
  image_size=224, keep_all=True, device=device)

class_map = {'Komolafe': 0, 'Mac': 1, 'Adedayo': 2,
             'Yisau': 3,'Cynthia': 4, 'Joseph': 5,
             'Michael': 6,'Jason': 7,'Ogechi': 8, 
             'Emmanuel': 9,'Mubaraq': 10,'Nasir': 11}

resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(class_map)
).to(device)

resnet.load_state_dict(torch.load('trained_resnet_state_dict.pth', map_location=device))

resnet.eval()


cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        cropped_image = mtcnn(frame)
        outputs = resnet(cropped_image.to(device))
        _, preds = torch.max(outputs, 1)
        pred_class = get_key(preds.item(), class_map)

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{pred_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Real-Time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



from facenet_pytorch import MTCNN, InceptionResnetV1
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
from itertools import combinations
import random
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

val_transform = A.Compose([
    A.HorizontalFlip(),
    ToTensorV2()
])

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, img1, img2, img3):
        # Pass each image through the Resnet
        emb_anchor = self.resnet(img1)
        emb_positive = self.resnet(img2)
        emb_negative = self.resnet(img3)
        return emb_anchor, emb_positive, emb_negative

def process_image(image_path, transform, label=None):
    """Load an image from the disk and apply transformations."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = transform(image=img)['image']  # Assuming you are using Albumentations for transforms
    img = img / 255.0
    if label:
        return img, label
    else:
        return img
    
def loop_classes(dir, model, neg_img):
    classlist = os.listdir(dir)
    results = []
    for class_name in classlist:
        imagepaths = os.listdir(os.path.join(dir, class_name))
        random.shuffle(imagepaths)
        ancpospair = [imagepath for imagepath in combinations(imagepaths[:5],2)]
        anc_path = os.path.join(dir, class_name, random.choice(ancpospair)[0])
        pos_path = os.path.join(dir, class_name, random.choice(ancpospair)[1])

        pred = infer_siamese(model, anc_path, pos_path, neg_img, val_transform, class_name)
        print(f"ClassName: {class_name}, PRED LABEL: {pred}")
        if pred == class_name:
            results.append(pred)

    return results
        
    
def get_labels(positive_distance, negative_distance, positive_label, threshold=0.5):
    if positive_distance <= threshold and negative_distance <= threshold:
        return positive_label
    elif positive_distance <= threshold and negative_distance > threshold:
        return "Uncorrelated"
    elif positive_distance > threshold and negative_distance > threshold:
        return "No Match Found"
    elif positive_distance > threshold and negative_distance <= threshold:
        return "Match with Negative (Error Case)"
    else:
        return "Ambiguous Case"

def infer_siamese(model, anchor_path, positive_path, negative_image, transform, label):
    # Process images
    anchor_img, anchor_label = process_image(anchor_path, transform, label)
    anchor_img = anchor_img.unsqueeze(0).to(device)  # Add batch dimension
    
    positive_img, positive_label = process_image(positive_path, transform, label)
    positive_img = positive_img.unsqueeze(0).to(device)
    
    assert anchor_label == positive_label, "Anchor and Positive Image should belong to the same person."
    
    negative_img = negative_image
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
        
        # Calculate distances
        positive_distance = (anchor_emb - positive_emb).pow(2).sum().sqrt().item()  # Scalar value
        negative_distance = (anchor_emb - negative_emb).pow(2).sum().sqrt().item()  # Scalar value
        
    label = get_labels(positive_distance, negative_distance, positive_label)
    return label

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

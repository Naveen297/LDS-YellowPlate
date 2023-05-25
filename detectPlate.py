import cv2
import numpy as np
from PIL import Image
import pytesseract
import torch
from yolov5.models.yolo import Model


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model_yaml = ckpt['model'].yaml
    model_yaml['anchors'] = 3 # add 'anchors' key
    model = Model(model_yaml).to("cpu")
    model.load_state_dict(ckpt['model'].float().state_dict())
    model.eval()
    return model



def detect_license_plate(image_path, model):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Convert to tensor

    # Inference
    results = model(img_tensor)

    # Extract bounding boxes
    boxes = results.xyxy[0].cpu().detach().numpy()

    # Filter boxes with label corresponding to 'license plate' if you've trained for other objects too
    license_plates = [box for box in boxes if box[5] == 0]

    # Assuming 'license_plates' now contains boxes around license plates, extract these regions
    for i, (x1, y1, x2, y2, conf, label) in enumerate(license_plates):
        # Crop the license plate out of the original image
        license_plate_img = img[int(y1):int(y2), int(x1):int(x2)]

        # Convert image to grayscale
        gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to perform OCR on the extracted license plate
        result = pytesseract.image_to_string(Image.fromarray(gray), lang='eng',
                                             config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        print(f"License Plate Text {i + 1}: {result}")


if __name__ == "__main__":
    model_path = 'best.pt'
    image_path = "test/images/image_0145_jpg.rf.21055e2b49fe44a1631f5e4a1b8a1be9.jpg"
    model = load_model(model_path)
    detect_license_plate(image_path, model)

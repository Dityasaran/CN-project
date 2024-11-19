import cv2
import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image

# Load the COCO class names (91 classes)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Set the confidence threshold
confidence_threshold = 0.2

# Load the model with weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# Load the input image
image_path = "input.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: '{image_path}' not found or failed to load.")
    exit(1)

# Convert the image to RGB format (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the OpenCV image (NumPy array) to a PIL image
pil_image = Image.fromarray(image_rgb)

# Prepare the image for the model
transform = weights.transforms()
input_tensor = transform(pil_image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    detections = model(input_tensor)

# Get the detections from the output
detections = detections[0]

# Iterate through detections and draw bounding boxes
for i in range(len(detections['boxes'])):
    score = detections['scores'][i].item()
    if score > confidence_threshold:
        box = detections['boxes'][i].numpy().astype(int)
        class_id = int(detections['labels'][i].item())

        # Check if the class_id is within the valid range
        if class_id < len(COCO_CLASSES):
            class_name = COCO_CLASSES[class_id]
        else:
            class_name = "Unknown"

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print the detected class name and confidence
        print(f"Detected: {class_name} with confidence {score:.2f}")

# Resize the image for display (adjust the size as needed)
max_display_width = 1024
max_display_height = 768
height, width = image.shape[:2]

# Calculate scaling factor
scale_width = max_display_width / width
scale_height = max_display_height / height
scale = min(scale_width, scale_height)

# Resize image if it's larger than display size
if scale < 1.0:
    display_image = cv2.resize(image, (int(width * scale), int(height * scale)))
else:
    display_image = image

# Display the output image
cv2.imshow("Detections", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import argparse
from ultralytics import YOLO
import torch
import os

# Set up the argument parser
parser = argparse.ArgumentParser(description='Run YOLO model prediction.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model file.')
parser.add_argument('--img_dir', type=str, required=True, help='Image directory for prediction. YOLOv9 folder organizaiton.')
parser.add_argument('--save_img', type=bool, default=False, help='If to save images with predictions.')
parser.add_argument('--save_crop', type=bool, default=False, help='If to save detected instances cropped from images.')
parser.add_argument('--batch_size', type=int, default=30, help='Batch size for processing images.')
parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold for detection.')
parser.add_argument('--iou', type=float, default=0.7, help='IOU threshold for detection.')
parser.add_argument('--img_size', type=int, default=640, help='Image size for processing.')

args = parser.parse_args()

# Check if Apple silicon mps device is available
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple silicon mps device.")
# Check if Nvidia CUDA device is available
elif torch.cuda.is_available():
    device = "cuda"
    print("Using Nvidia CUDA device.")
# If no GPU is available, use CPU
else:
    device = "cpu"
    print("Using CPU.")

assert os.path.exists(args.model_path), f"Model path {args.model_path} does not exist."
model_exp_name = 'yolo' + args.model_path.split('yolo')[1].split('/')[0]
split = os.path.basename(os.path.dirname(args.img_dir))
exp_name = '_'.join(["predict", split, f'conf-{args.conf}', f'iou-{args.iou}', model_exp_name])

# Load pretrained model
model = YOLO(args.model_path)

# Specify predict data source
img_dir = args.img_dir
assert os.path.exists(img_dir), f"{img_dir} not found"
source = os.path.join(img_dir, '*.jpg')

# Use the model to predict
results = model.predict(source=source, name=exp_name, device=device, batch=args.batch_size, imgsz=args.img_size,
                        iou=args.iou, conf=args.conf, save_txt=True, save_conf=True, save=args.save_img, save_crop=args.save_crop)

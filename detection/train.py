import argparse
from ultralytics import YOLO
import torch
import os

# Setup argparse
# NOTE: increase batchsize if freezing backbone
parser = argparse.ArgumentParser(description='Train a YOLO model with custom parameters.')
parser.add_argument('--img_size', type=int, default=640, help='image size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--model_name', type=str, default='yolov8l', help='model name')
parser.add_argument('--batch_size', type=int, default=28, help='batch size')
parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold')
parser.add_argument('--optimizer', type=str, default='auto', help='optimizer')
parser.add_argument('--pretrain', type=str, default='coco', help='pretraining dataset')
parser.add_argument('--train_data', type=str, default='duke', help='training data set')
parser.add_argument('--data_config_path', type=str, default='/home/psc/Desktop/PSC/Custom_Dataset/custom-dataset-4120.v1i.yolov8/data.yaml', help='data config path')
parser.add_argument('--freeze_backbone', action='store_true', help='if freezing the backbone')
parser.add_argument('--fraction', type=float, default=1.0, help='the fraction of the dataset to use for training')
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

# Arguments
img_size = args.img_size
epochs = args.epochs
model_name = args.model_name
batch_size = args.batch_size
conf = args.conf
iou = args.iou
optimizer = args.optimizer
pretrain = args.pretrain
train_data = args.train_data
freeze_backbone = args.freeze_backbone
fraction = args.fraction

# backbone layers: 0-9
freeze_layer = 10 if freeze_backbone else None

exp_name = '_'.join([model_name, f'imgsz-{str(img_size)}', f'{epochs}-epochs', f'batch-{batch_size}', 
                     f'conf-{conf}', f'iou-{iou}', f'optimizer-{optimizer}', 
                     f'pretrain-{pretrain}', f'train_data-{train_data}', f'fraction-{fraction}',
                     f'freeze-backbone-{freeze_backbone}'])
# exp_name = 'temp'


# if train_data == 'duke':
#     data_config_path = '/home/psc/Desktop/PSC/Duke_Dataset/duke-solar.v4i.yolov8/data.yaml'
# elif train_data == 'custom1000':
#     data_config_path = '/home/psc/Desktop/PSC/Custom_Dataset/custom-dataset-4120.v1i.yolov8/data.yaml'
data_config_path = args.data_config_path
assert os.path.exists(data_config_path), f"{data_config_path} not found"

# Load a model
model = YOLO(f"{model_name}.yaml")  # load model architecture
if pretrain == 'coco':
    try:
        model = model.load(f"{model_name.split('-')[0]}.pt")  # load a pretrained model (recommended for training)
        print(f"Train {model_name} using weight pretrained on coco")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
elif pretrain == 'duke':
    model = model.load('/home/psc/Desktop/PSC/tools/runs/detect/yolov8l_imgsz-500_70-epochs_batch-26_conf-0.5_iou-0.5_optimizer-auto_pretrain-coco_train_data-duke_freeze-backbone-False/weights/best.pt')  # load a pretrained model (recommended for training)
    print(f"Train {model_name} using weight pretrained on duke dataset")
else:
    print(f"Train {model_name} from scratch")

# Use the model
model.train(data=data_config_path, fraction=fraction, name=exp_name, epochs=epochs, device=[0,1], plots=True, 
            batch=batch_size, imgsz=img_size, iou=iou, conf=conf, 
            optimizer=optimizer, save_period=10, freeze=freeze_layer)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# path = model.export(format="onnx")  # export the model to ONNX format
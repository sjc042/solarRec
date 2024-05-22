from data_retrieval import query_from_API
from detection_and_postprocess import run_detection_and_analysis
from gridpoint_generation import generate_grid

import os

def test1_ProcessCSV():
    csv_path = '/home/psc/Desktop/solarRec/data/test_data/TestAddresses.csv'
    src_dir = os.path.dirname(csv_path)
    query_from_API(csv_path, save_dir=src_dir)
    model_path = '/home/psc/Desktop/solarRec/detection/checkpoints/yolov8l-seg_imgsz-640_100-epochs_batch-28_conf-0.5_iou-0.5_optimizer-auto_pretrain-coco_train_data-seg2000.pt'
    run_detection_and_analysis(src_dir, model_path, save_img=True)

def test2_addressList():
    address_list = ['4125 187th Ave SE, Issaquah, WA 98027', '510 pays rd, Cle Elum, WA']
    save_dir = '/home/psc/Desktop/solarRec/data/test_list'
    # query_from_API(address_list, save_dir=save_dir)
    model_path = '/home/psc/Desktop/solarRec/detection/checkpoints/yolov8l-seg_imgsz-640_100-epochs_batch-28_conf-0.5_iou-0.5_optimizer-auto_pretrain-coco_train_data-seg2000.pt'
    run_detection_and_analysis(save_dir, model_path, save_img=True, conf=0.1)

def test3_addressTXT():
    txt_file = '/home/psc/Desktop/solarRec/data/demo_data/addresses.txt'
    save_dir = os.path.dirname(txt_file)
    # query_from_API(txt_file, save_dir=save_dir)
    model_path = '/home/psc/Desktop/solarRec/detection/checkpoints/yolov8l-seg_imgsz-640_100-epochs_batch-28_conf-0.5_iou-0.5_optimizer-auto_pretrain-coco_train_data-seg2000.pt'
    run_detection_and_analysis(save_dir, model_path, save_img=True)

def test4_coordinateGrid():
    height = 5
    width = 5
    # coordinate_list = generate_grid((47.6326, -122.2133), height, width)
    # save_dir = '/home/psc/Desktop/solarRec/data/test_coords'
    save_dir = r"D:\solarRec\solarRec\data\test_grid"
    # query_from_API(coordinate_list, save_dir=save_dir)
    model_path = r"D:\solarRec\solarRec\detection\checkpoints\yolov8l-seg_imgsz-640_100-epochs_batch-28_conf-0.5_iou-0.5_optimizer-auto_pretrain-coco_train_data-seg2000.pt"
    run_detection_and_analysis(save_dir, model_path, save_img=True, conf=0.1, process_grid=True)

def main():
    test4_coordinateGrid()

if __name__ == "__main__":
    main()
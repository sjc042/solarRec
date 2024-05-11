from data_retrieval import query_from_API
from detection_and_postprocess import run_detection_and_analysis

import os

def test1_ProcessCSV():
    csv_path = '/home/psc/Desktop/solarRec/data/test_data2/TestAddresses.csv'
    src_dir = os.path.dirname(csv_path)
    # query_from_API(csv_path, save_dir=src_dir)
    model_path = '/home/psc/Desktop/solarRec/detection/checkpoints/yolov8l-seg_imgsz-640_100-epochs_batch-28_conf-0.5_iou-0.5_optimizer-auto_pretrain-coco_train_data-seg2000.pt'
    run_detection_and_analysis(src_dir, model_path, save_img=True)


def main():
    test1_ProcessCSV()

if __name__ == "__main__":
    main()
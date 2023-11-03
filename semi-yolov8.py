from ultralytics import YOLO

    
if __name__ == '__main__':
    # Parameter
    pretrain_weight = 'runs/detect/yolov8l_base_1280_batch_4_agent3/weights/best.pt'
    imgsz = 1280
    batch_size = 4

    name = 'yolov8l_semi_' + str(imgsz) + '_batch_' + str(batch_size) + '_agent'

    # Training.
    model = YOLO(model=pretrain_weight)

    results = model.train(
        data = 'configs/semi_yolov8.yaml',
        imgsz = imgsz,
        # rect = True, # if input not square set this true
        mosaic = True,
        device = '4',
        epochs = 10,
        optimizer='SGD',
        lr0=0.0001,
        batch = batch_size,
        # resume=True,
        name = name
    )
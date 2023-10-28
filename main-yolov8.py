from ultralytics import YOLO

    
if __name__ == '__main__':
    # Parameter
    pretrain_weight = 'yolov8l.pt'
    imgsz = 1280
    batch_size = 4

    name = pretrain_weight.split(".")[0] + '_base_' + str(imgsz) + '_batch_' + str(batch_size) + '_agent'

    # Training.
    model = YOLO(model=pretrain_weight)

    results = model.train(
        data = 'configs/base_yolov8.yaml',
        imgsz = imgsz,
        # rect = True, # if input not square set this true
        mosaic = True,
        device = '2',
        epochs = 50,
        optimizer='SGD',
        lr0=0.001,
        batch = batch_size,
        # resume=True,
        name = name
    )
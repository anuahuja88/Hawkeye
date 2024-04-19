
from ultralytics import YOLO

model = YOLO('yolov9c.pt')

results = model.train(data = 'Hawkeye/Hawkeye.v1i.yolov9/data.yaml', epochs = 100, batch = -1)
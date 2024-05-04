
from ultralytics import YOLO

model = YOLO('yolov9c.pt')

results = model.train(data = '/Users/anuahuja/Desktop/2024/Computer Vision/Hawkeye/Hawkeye.v3i.yolov9/data.yaml', epochs = 100, batch = -1)
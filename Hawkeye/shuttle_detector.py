
from ultralytics import YOLO

model = YOLO('yolov9c.pt')

results = model.train(data = '/csse/users/aah109/Desktop/Hawkeye/Hawkeye.v3i.yolov9/data.yaml', epochs = 100, batch = -1)

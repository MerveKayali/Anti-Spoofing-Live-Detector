from ultralytics import YOLO
import torch
import os

model = YOLO("yolov8l.pt")

def main():
    model.train(data=r"C:\Users\merve\PycharmProjects\pythonProject\AntiSpoofing\Dataset\SplitData\dataOffline.yaml",epochs=300)

if __name__ =='__main__':
    main()

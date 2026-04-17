from ultralytics import YOLO

# # Load a model
image_size = 192

model = YOLO("exp-2.pt")  

model.export(format="tflite", imgsz = image_size, int8 = True, data="coco128.yaml")# change coco128 to sth real later


#model.export(format="onnx", imgsz=192)
# pip install -U onnx2tf   pip install -U h5py==3.7.0   pip install -U psutil==5.9.5   pip install -U ml_dtypes==0.2.0
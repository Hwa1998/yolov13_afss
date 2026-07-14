import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

if __name__ == '__main__':
    # model.export(format='onnx', simplify=True, opset=11, imgsz=[320, 480])  # [h, w]
    # model.export(format='onnx', simplify=True, opset=11, imgsz=[640, 3840])
    # model.export(format='onnx', simplify=True, opset=11, imgsz=[512, 512])
    model = YOLO(r'best.pt')
    model.export(format='onnx', simplify=False, opset=12, imgsz=[640, 640], dynamic=False)
    # model.export(format='onnx', simplify=True, opset=12, imgsz=[2624, 672], dynamic=False)
    
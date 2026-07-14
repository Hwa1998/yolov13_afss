'''
这是 tensorrt 8.6.1.16 版本的目标检测推理代码
并且分别支持了 detect+nmsfree、v10 detect 这两种检测头，分别对应的是 postprocess 和 postprocess_v10detect 这两种后处理
'''


import numpy as np
import time
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA上下文
import os


# # coco80个类别
# CLASSES = {0: 'right_1', 1: 'right_2', 2: 'right_3', 3: 'right_4', 4: 'left_5', 5: 'left_6', 6: 'left_7'}




class YOLOv8TRT:
    def __init__(self, engine_path=None, CLASSES=None):
        """初始化YOLOv8 TensorRT检测器

        Args:
            engine_path (str, optional): TensorRT引擎文件路径. 默认为None.
        """
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.stream = None
        self.classes = CLASSES
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self._allocated_buffers = []  # 跟踪所有分配的设备内存
        self.engine_data = None  # 存储引擎数据


        if engine_path:
            self.load_model(engine_path)

    def load_model(self, engine_path):
        """加载TensorRT引擎 (TensorRT 10.8)"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            # 读取引擎文件数据
            f = open(engine_path, "rb")
            self.engine_data = f.read()
            f.close()

            self.engine = runtime.deserialize_cuda_engine(self.engine_data)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self._allocate_buffers()  # 使用新的内存分配方法

            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _allocate_buffers(self):
        """为所有输入和输出张量分配缓冲区 (TensorRT 8.6 绑定API)"""
        self.inputs = []
        self.outputs = []
        self.bindings = []  # 用于存储设备内存地址列表
        self._allocated_buffers = []

        # TensorRT 8.6 使用绑定索引而非张量名称
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)

            # 处理动态批次维度
            if shape[0] == -1:
                shape[0] = 1  # 设置为默认批次大小

            # 计算需要分配的内存大小
            volume = abs(trt.volume(shape))
            size = volume * dtype.itemsize

            # 分配设备内存
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            self._allocated_buffers.append(device_mem)

            # 确定是输入还是输出
            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'index': i,
                    'name': name,
                    'shape': shape,
                    'dtype': np.dtype(trt.nptype(dtype)),
                    'device_mem': device_mem
                })
            else:
                self.outputs.append({
                    'index': i,
                    'name': name,
                    'shape': shape,
                    'dtype': np.dtype(trt.nptype(dtype)),
                    'device_mem': device_mem
                })

        # 调试信息
        print(f"Allocated buffers for {len(self.inputs)} input(s) and {len(self.outputs)} output(s).")
        for inp in self.inputs:
            print(f"Input: name={inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")
        for out in self.outputs:
            print(f"Output: name={out['name']}, shape={out['shape']}, dtype={out['dtype']}")

    def cleanup(self):
        """清理所有分配的资源"""
        print("Cleaning up resources...")

        # 释放所有分配的设备内存
        for buf in self._allocated_buffers:
            if buf:
                buf.free()

        self._allocated_buffers = []
        self.inputs = []
        self.outputs = []
        self.bindings = []  # 清空绑定列表

        # 释放引擎数据
        self.engine_data = None

        # 释放上下文和引擎
        if self.context:
            self.context = None

        if self.engine:
            self.engine = None

        print("Resources cleaned up successfully.")
    def warmup(self, warmup_iterations=3):
        """模型热身，使用随机数据进行几次推理以初始化模型

        Args:
            warmup_iterations (int): 热身迭代次数
        """
        if self.engine is None:
            raise ValueError("模型未加载，请先调用 load_model 方法")

        print("开始 tensorrt 模型热身...")
        warmup_start = time.time()

        # 创建与模型输入尺寸一致的随机数据 (1, 3, 640, 640)
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        for i in range(warmup_iterations):
            self.inference(dummy_input)
            print(f"tensorrt 热身迭代 {i + 1}/{warmup_iterations} 完成")

        warmup_end = time.time()
        print(f"tensorrt 模型热身完成，耗时: {warmup_end - warmup_start:.4f} 秒")

    def preprocess(self, image):
        """预处理输入图像

        Args:
            image (np.ndarray): 输入图像，BGR格式

        Returns:
            tuple: (预处理后的blob, 缩放比例)
        """
        if image is None:
            raise ValueError("输入图像为空")
        [height, width, _] = image.shape
        # 预处理
        scale = min(640/height, 640/width)
        resize_h = int(height*scale)
        resize_w = int(width*scale)
        resize_img = cv2.resize(image, (resize_w, resize_h))
        padding_h = int((640 - resize_h) / 2)
        padding_w = int((640 - resize_w) / 2)

        img = np.ones((640, 640, 3), dtype=np.uint8) * 114
        img[padding_h:padding_h+resize_h, padding_w:padding_w+resize_w, :] = resize_img
        padded_image = img

        # 归一化、等比缩放、转换通道顺序 (HWC to CHW)
        blob = cv2.dnn.blobFromImage(padded_image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        blob = np.ascontiguousarray(blob)

        return blob, scale, resize_h, resize_w

    def inference(self, blob):
        """执行模型推理 (TensorRT 8.6 绑定API)"""
        if self.engine is None:
            raise ValueError("模型未加载，请先调用 load_model 方法")

        if not self.inputs:
            raise RuntimeError("No input bindings found.")

        # 获取输入绑定信息
        input_binding = self.inputs[0]

        # 处理动态输入形状 (如果支持)
        if -1 in input_binding['shape']:
            # TensorRT 8.6 设置动态形状的方式
            profile = self.engine.get_profile_shape(0, input_binding['name'])
            # 设置优化配置文件
            self.context.set_optimization_profile_async(0, self.stream.handle)
            # 设置实际的输入形状
            self.context.set_binding_shape(input_binding['index'], blob.shape)

        # 将输入数据复制到设备
        cuda.memcpy_htod_async(input_binding['device_mem'], blob, self.stream)

        # 执行推理 - 使用 execute_async_v2
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # 处理输出
        outputs = []
        for out_info in self.outputs:
            # 获取实际的输出形状
            if self.context.all_binding_shapes_specified:
                actual_shape = self.context.get_binding_shape(out_info['index'])
            else:
                actual_shape = out_info['shape']

            # 创建主机内存并复制数据
            host_mem = np.empty(actual_shape, dtype=out_info['dtype'])
            cuda.memcpy_dtoh_async(host_mem, out_info['device_mem'], self.stream)
            outputs.append(host_mem)

        self.stream.synchronize()

        if not outputs:
            raise RuntimeError("No outputs were processed.")
        # 这里用 outputs[0] 是因为，nmsfree 的代码，在 tensorrt 10.8 的版本中，转 engine 时，是最大的 output shape 的那个在最前面的
        #                                       在 tensorrt 8.6.1.16 的版本中，转 engine 时，是最大的 output shape 的那个在最后面的
        # return outputs[0]
        return outputs[-1]
    # 适配 yolov8、yolov13的 head就是继承代码中调用  detect 和 nmsfree 的这两个检测头
    def postprocess(self, outputs, scale, resize_h, resize_w, confidence_threshold=0.25, nms_threshold=0.45):
        """优化后的后处理函数 - 使用向量化操作

        Args:
            outputs: 模型输出
            scale: 缩放比例
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值

        Returns:
            list: 检测结果列表，每个元素为 (class_id, confidence, box)
                  其中 box 为 [x, y, width, height]
        """
        # 转换输出形状: [1, 84, 8400] -> [8400, 84]
        if outputs.shape[0] == 1:
            outputs = outputs[0].T
        else:
            outputs = outputs.T

        # 提取框坐标 (cx, cy, w, h)
        boxes_xywh = outputs[:, :4].copy()

        # 提取类别置信度
        classes_scores = outputs[:, 4:]

        # 找到每个框的最大类别分数和对应的类别ID
        max_scores = np.max(classes_scores, axis=1)
        class_ids = np.argmax(classes_scores, axis=1)

        # 应用置信度阈值过滤
        keep = max_scores > confidence_threshold

        # 如果没有检测到任何目标，直接返回空列表
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        max_scores = max_scores[keep]
        class_ids = class_ids[keep]

        # 转换坐标格式: (cx, cy, w, h) -> (x, y, w, h)
        boxes_xywh[:, 0] -= boxes_xywh[:, 2] / 2  # x = cx - w/2
        boxes_xywh[:, 1] -= boxes_xywh[:, 3] / 2  # y = cy - h/2

        # 应用NMS - 使用预分配的数组避免重复转换
        boxes_list = boxes_xywh.tolist()
        scores_list = max_scores.tolist()

        nms_keep = cv2.dnn.NMSBoxes(
            boxes_list, scores_list,
            confidence_threshold, nms_threshold
        )

        if len(nms_keep) == 0:
            return []

        # 预分配结果列表
        n_boxes = len(nms_keep)
        detections = [None] * n_boxes

        # 填充结果
        for i, idx in enumerate(nms_keep.flatten()):
            x, y, w, h = boxes_xywh[idx]

            # 将坐标映射回原图尺寸
            padding_h = int((640 - resize_h) / 2)
            padding_w = int((640 - resize_w) / 2)

            x = int((x - padding_w)/ scale)
            y = int((y - padding_h) / scale)
            w = int(w / scale)
            h = int(h / scale)

            detections[i] = {
                'class_id': int(class_ids[idx]),
                'confidence': float(max_scores[idx]),
                'box': [x, y, w, h],
                'class_name': self.classes[int(class_ids[idx])]
            }

        return detections
    # 适配 v10detect 这个检测头
    def postprocess_v10detect(self, outputs, scale, resize_h, resize_w, confidence_threshold=0.25, nms_threshold=0.45):
        """向量化版本的后处理函数 - 更高效

        Args:
            outputs: 模型输出，形状为 (1, 300, 6)
            scale: 缩放比例
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值

        Returns:
            list: 检测结果列表
        """
        # 输出形状: (1, 300, 6) -> 直接取第一个batch
        predictions = outputs[0]  # (300, 6)

        # 提取各个分量
        boxes_xywh = predictions[:, :4]  # (300, 4) - [x, y, w, h]
        confidences = predictions[:, 4]  # (300,) - confidence
        class_ids = predictions[:, 5]  # (300,) - class_id

        # 应用置信度阈值
        keep = confidences > confidence_threshold

        if not np.any(keep):
            return []

        # 过滤低置信度预测
        boxes_xywh = boxes_xywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # 将中心坐标转换为左上角坐标
        boxes_xyxy = boxes_xywh.copy()

        # 应用NMS
        boxes_list = boxes_xyxy.tolist()
        scores_list = confidences.tolist()

        nms_keep = cv2.dnn.NMSBoxes(
            boxes_list, scores_list,
            confidence_threshold, nms_threshold
        )

        if len(nms_keep) == 0:
            return []

        # 构建最终结果
        detections = []
        for idx in nms_keep.flatten():
            x1, y1, x2, y2 = boxes_xyxy[idx]
            print(x1, y1, x2, y2)
            padding_h = int((640 - resize_h) / 2)
            padding_w = int((640 - resize_w) / 2)
            x = int((x1 - padding_w)) / scale
            y = int((y1 - padding_h)) / scale
            w = int(x2-x1) / scale
            h = int(y2-y1) / scale
            detections.append({
                'class_id': int(class_ids[idx]),
                'confidence': float(confidences[idx]),
                'box': [int(x), int(y), int(w), int(h)],
                'class_name': self.classes[int(class_ids[idx])]
            })

        return detections

    def predict(self, image, confidence_threshold=0.25, nms_threshold=0.45):
        blob, scale, resize_h, resize_w = self.preprocess(image)
        outputs = self.inference(blob)
        detections = self.postprocess(outputs, scale, resize_h, resize_w, confidence_threshold, nms_threshold)
        # detections = self.postprocess_v10detect(outputs, scale, resize_h, resize_w, confidence_threshold, nms_threshold)
        # image_ = self.draw_detections(image, detections)
        return detections

    def draw_detections(self, image, detections, fps_text=""):
        """在图像上绘制检测结果和FPS

        Args:
            image (np.ndarray): 输入图像
            detections (list): 检测结果列表
            fps_text (str): 要显示的FPS文本

        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        result_image = image.copy()

        # 在图像左上角显示FPS
        if fps_text:
            cv2.putText(result_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for detection in detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            box = detection['box']

            x, y, width, height = box
            x_plus_w = x + width
            y_plus_h = y + height

            label = f'{self.classes[class_id]} ({confidence:.2f})'
            color = self.colors[class_id]

            # 绘制矩形框
            cv2.rectangle(result_image, (x, y), (x_plus_w, y_plus_h), color, 2)
            # 绘制类别
            cv2.putText(result_image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_image


if __name__ == '__main__':
    engine_path = r'D:\codes\deep_learning\yolov13-main\runs\nmsfree\bms_c37_yolov8_nmsfree\weights\best.engine'
    classes = {0: 'bms'}
    images_path = r'D:\datasets\yolo\detect\bms__for_c37_3\train\images'
    """图片推理"""
    detector = YOLOv8TRT(engine_path=engine_path, CLASSES=classes)


    # 热身
    detector.warmup(1)

    images_name_list = os.listdir(images_path)
    for image_name in images_name_list:
        image_path = os.path.join(images_path, image_name)
        # 读取图片
        image = cv2.imread(image_path)

        # 推理
        start_time = time.time()
        detections = detector.predict(image)
        inference_time = time.time() - start_time

        # 显示结果
        print(f"推理时间: {inference_time * 1000:.2f} ms")
        print(f"检测到 {len(detections)} 个目标:")

        for detection in detections:
            print(f"  {detection['class_name']}: {detection['confidence']:.3f}")

        # 绘制并显示
        result_image = detector.draw_detections(image, detections, f"FPS: {1 / inference_time:.1f}")
        cv2.imshow('Single Image Test', result_image)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    detector.cleanup()
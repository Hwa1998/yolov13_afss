from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["WANDB_MODE"] = "offline"
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # train
    model = YOLO('cfg/models/v13/yolov13.yaml').load(r'D:\DingReihwa\6_AI\codes\yolov13-main\yolov13n.pt')
    # Train the model
    # 数据预处理的过程
    #   1、等比 resize：imgsz=640，image_shape=[1024, 1280]---image_new_shape=[512, 640]
    #   2、padding 成 32 的整数倍，就算是resize了以后已经是 32 的整数倍了也还要 padding 一次---image_new_shape=[544, 672]
    #   3、输入到神经网络的尺寸就是 [h, w]=[544, 672]
    #   4、在导出 onnx 或者是推理的时候，只要写死 [512, 640] 的输入就好，然后让自适应缩放去做处理，这样就能保证推理结果一致了
    # workers = 8，这个在 windows 下是能用的，只要放到 main 函数里面就好了
    #   https://blog.csdn.net/m0_50617544/article/details/121441585
    model.train(data='cfg/datasets/VOC.yaml',
                epochs=3000, imgsz=[640, 640], batch=16, workers=12, lr0=0.003, 
                afss=True, afss_easy_thresh=0.95, afss_hard_thresh=0.2, afss_moderate_ratio=0.7, afss_update_interval=5, afss_warmup_epochs=30, afss_auto_tune=False,
                name='save')

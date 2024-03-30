import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation

def object_detection_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    segment_image = instance_segmentation()
    segment_image.load_model(r"C:\Users\propr\PycharmProjects\pythonProject52\mask_rcnn_coco_tf.h5")
    target_class = segment_image.select_target_classes(person=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = segment_image.segmentFrame(
            frame,
            segment_target_classes=target_class
        )

        output = result[1]
        cv2.imshow('Object Segmentation', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажмите 'q' для выхода из видео
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "file1.mp4"  # Путь к видео для обработки
    object_detection_on_video(video_path)

if __name__ == '__main__':
    main()

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', r'C:\Users\propr\PycharmProjects\pythonProject52\file1.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

# Ошибка:
# C:\Users\propr\PycharmProjects\pythonProject52\.venv\Scripts\python.exe C:\Users\propr\PycharmProjects\pythonProject52\Variant1.py
# WARNING:tensorflow:From C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\mask_rcnn.py:31: The name tf.disable_eager_execution is deprecated. Please use tf.compat.v1.disable_eager_execution instead.
#
# WARNING:tensorflow:From C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\tensorflow\python\util\deprecation.py:660: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
# Instructions for updating:
# Use fn_output_signature instead
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# W0000 00:00:1711821332.800322    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 256 } dim { size: 256 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -2 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -2 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -2 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821332.804248    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 128 } dim { size: 128 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -5 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -5 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -5 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821332.804322    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 64 } dim { size: 64 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -7 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -7 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -7 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821332.804407    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 32 } dim { size: 32 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -9 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -9 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -9 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821333.128183    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 256 } dim { size: 256 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -17 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -17 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -17 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# W0000 00:00:1711821333.128481    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 128 } dim { size: 128 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -19 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -19 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -19 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# W0000 00:00:1711821333.128547    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 64 } dim { size: 64 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -21 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -21 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -21 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# W0000 00:00:1711821333.128626    6472 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 32 } dim { size: 32 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -23 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -23 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -23 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\utils.py:566: FutureWarning: In the future `np.bool` will be defined as the corresponding NumPy scalar.
#   mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
# Traceback (most recent call last):
#   File "C:\Users\propr\PycharmProjects\pythonProject52\Variant1.py", line 36, in <module>
#     main()
#   File "C:\Users\propr\PycharmProjects\pythonProject52\Variant1.py", line 33, in main
#     object_detection_on_video(video_path)
#   File "C:\Users\propr\PycharmProjects\pythonProject52\Variant1.py", line 17, in object_detection_on_video
#     result = segment_image.segmentFrame(
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\__init__.py", line 442, in segmentFrame
#     results = self.model.detect([new_frame])
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\mask_rcnn.py", line 2470, in detect
#     self.unmold_detections(detections[i], mrcnn_mask[i],
#   File "C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\mask_rcnn.py", line 2417, in unmold_detections
#     full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\utils.py", line 566, in unmold_mask
#     mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
#                                                     ^^^^^^^
#   File "C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\numpy\__init__.py", line 338, in __getattr__
#     raise AttributeError(__former_attrs__[attr])
# AttributeError: module 'numpy' has no attribute 'bool'.
# `np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
# The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
#     https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'bool_'?
#
# Process finished with exit code 1
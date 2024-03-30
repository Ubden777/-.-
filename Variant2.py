import cv2
import numpy as np
import os
from pixellib.instance import (instance_segmentation)
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
video_path = "file1.mp4"
cap = cv2.VideoCapture(video_path)

segment_image = instance_segmentation()
segment_image.load_model(r"C:\Users\propr\PycharmProjects\pythonProject52\mask_rcnn_coco.h5")
# segment_image.load_model(r"C:\Users\propr\PycharmProjects\pythonProject52\mask_rcnn_coco.h5")
target_class = segment_image.select_target_classes(person=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = segment_image.segmentFrame(frame, segment_target_classes=target_class)

    output = result[1]
    cv2.imshow('Object Segmentation', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажмите 'q' для выхода из видео
        break

cap.release()
cv2.destroyAllWindows()

# Ошибка
# C:\Users\propr\PycharmProjects\pythonProject52\.venv\Scripts\python.exe C:\Users\propr\PycharmProjects\pythonProject52\111.py
# 2024-03-31 01:00:11.706197: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-03-31 01:00:12.448843: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\mask_rcnn.py:31: The name tf.disable_eager_execution is deprecated. Please use tf.compat.v1.disable_eager_execution instead.
#
# 2024-03-31 01:00:14.606311: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING:tensorflow:From C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\tensorflow\python\util\deprecation.py:660: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
# Instructions for updating:
# Use fn_output_signature instead
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# W0000 00:00:1711821679.868853   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 256 } dim { size: 256 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -2 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -2 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -2 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821679.869479   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 128 } dim { size: 128 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -5 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -5 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -5 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821679.869544   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 64 } dim { size: 64 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -7 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -7 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -7 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821679.869622   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 32 } dim { size: 32 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -9 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -9 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 7 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -9 } dim { size: 7 } dim { size: 7 } dim { size: 256 } } }
# W0000 00:00:1711821680.203428   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 256 } dim { size: 256 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -17 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -17 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -17 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# W0000 00:00:1711821680.203609   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 128 } dim { size: 128 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -19 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -19 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -19 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# W0000 00:00:1711821680.203673   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 64 } dim { size: 64 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -21 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -21 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -21 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# W0000 00:00:1711821680.203752   22716 op_level_cost_estimator.cc:699] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_FLOAT } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_FLOAT shape { dim { size: 1 } dim { size: 32 } dim { size: 32 } dim { size: 256 } } } inputs { dtype: DT_FLOAT shape { dim { size: -23 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -23 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 14 } } device { type: "CPU" vendor: "AuthenticAMD" model: "248" frequency: 2096 num_cores: 12 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 524288 l3_cache_size: 8388608 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -23 } dim { size: 14 } dim { size: 14 } dim { size: 256 } } }
# C:\Users\propr\PycharmProjects\pythonProject52\.venv\Lib\site-packages\pixellib\instance\utils.py:566: FutureWarning: In the future `np.bool` will be defined as the corresponding NumPy scalar.
#   mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
# Traceback (most recent call last):
#   File "C:\Users\propr\PycharmProjects\pythonProject52\111.py", line 22, in <module>
#     result = segment_image.segmentFrame(frame, segment_target_classes=target_class)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
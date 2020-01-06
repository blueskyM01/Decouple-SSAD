import tensorflow as tf
import numpy as np

def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"
    就是计算两个时间段的IOU

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])

span_A = [0.231, 0.1379] # center
span_B = [0.16210938, 0.30078125]

span_A_ = [span_A[0] - span_A[1] / 2, span_A[0] + span_A[1] / 2]
print('span_A_:', span_A_)
iou = temporal_iou(span_A_, span_B)

print(iou)
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

def mx_GetFuseAndUnfuseIdx(fuse_idx, anchor_original):
    '''
    Introduction: get final idx: unfused anchor idx + fused anchor idx
    :param fuse_idx: 融合的anchor的位置
    :param anchor_original: type: list, original anchor before fuse
    :return:
            final_fuse: get final idx: unfused anchor idx + fused anchor idx
                        for example: [0, [1,2], 3, [4,5],6,7,8,9] (num of anchor_original is 10, so it starts from 0 and end at 9)
    '''
    final_fuse = fuse_idx.copy()
    if len(fuse_idx) != 0:
        # 补全连续
        count = 0
        for idx in range(len(fuse_idx) - 1):
            current_f = fuse_idx[idx]
            next_f = fuse_idx[idx + 1]
            if next_f[0] - current_f[1] >= 2:

                for i in range(next_f[0] - current_f[1] - 1):
                    final_fuse.insert(idx + i + count + 1, current_f[1] + i + 1)
                count = count + 1

        # 补全两端
        if isinstance(final_fuse[0], list):
            if final_fuse[0][0] > 0:
                for i in range(final_fuse[0][0]):
                    final_fuse.insert(i, i)
        else:
            if final_fuse[0] > 0:
                for i in range(final_fuse[0]):
                    final_fuse.insert(i, i)

        if isinstance(final_fuse[-1], list):
            if final_fuse[-1][1] < len(anchor_original) - 1:
                a = final_fuse[-1][1]
                for i in range(len(anchor_original) - a - 1):
                    final_fuse.insert(len(final_fuse), a + i + 1)
        else:
            if final_fuse[-1] < len(anchor_original) - 1:
                b = final_fuse[-1]
                for i in range(len(anchor_original) - b - 1):
                    final_fuse.insert(len(final_fuse), b + i + 1)
    else:
        final_fuse = []
        for i in range(len(anchor_original)):
            final_fuse.append(i)
    final_fuse_list = []
    for ele in final_fuse:
        if isinstance(ele, list):
            final_fuse_list.append(ele)
        else:
            final_fuse_list.append([ele, ele])

    final_fuse = np.array(final_fuse_list, dtype=np.int32)

    return final_fuse

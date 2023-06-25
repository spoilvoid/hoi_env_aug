# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import numpy as np
from collections import defaultdict

class HICOEvaluator():
    def __init__(self, preds, gts, subject_category_id, rare_triplets, non_rare_triplets, correct_mat):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []
        
        self.true_hoi_details = []
        self.false_hoi_details = []

        self.preds = []
        # preds是所有图像数据集的所有predictions，我们每次从中取出一个图像的predictions，称为img_preds
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T
            
            # ravel()函数的作用是将数据拉成一维数组
            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            # 存在hoi时
            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            self.preds.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        self.gts = []
        for img_gts in gts:
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id'}
            self.gts.append({
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in img_gts['hois']]
            })
            # 对于刚刚放入gts中的注释其中的‘hoi_annotation’部分的每一条记录
            for hoi in self.gts[-1]['hoi_annotation']:
                # 两个bbox对应的类id和hoi_id
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1

    def evaluate(self):
        # 对一张图片的pred和gts
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                # 如果gt_bboxes没有的话，直接所有预测的hoi全部判错
                hoi_details_for_one_image = []
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    hoi_detail = {}
                    hoi_detail['sub_box'] = pred_bboxes[pred_hoi['subject_id']]
                    hoi_detail['obj_box'] = pred_bboxes[pred_hoi['object_id']]
                    hoi_detail['category_id'] = pred_hoi['category_id']
                    hoi_detail['score'] = pred_hoi['score']
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    hoi_details_for_one_image.append(hoi_detail)
                    self.score[triplet].append(pred_hoi['score'])
                self.false_hoi_details.append(hoi_details_for_one_image)
                self.true_hoi_details.append([])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        # keys是pred的bbox的所有下标
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        # 对pred_hois四元组，按照score从大到小排序，如果没有score项，则按0进行排序
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            true_hoi_details_for_one_image = []
            false_hoi_details_for_one_image = []
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    # 这里的id指的都是gts的bbox下标
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    # 在这里选出两overlap中最小值最大的可匹配的gt_hoi的下标与对应的overlap
                    for gt_hoi in gt_hois:
                        # 若预测的hoi与gt成功匹配上
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                # triplet存储sub、obj、hoi三者的类id
                triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                hoi_detail = {}
                hoi_detail['sub_box'] = pred_bboxes[pred_hoi['subject_id']]
                hoi_detail['obj_box'] = pred_bboxes[pred_hoi['object_id']]
                hoi_detail['category_id'] = pred_hoi['category_id']
                hoi_detail['score'] = pred_hoi['score']
                # 防止判断出错，前面通过照理说应该能在gt_triplets中找到
                if triplet not in self.gt_triplets:
                    continue
                # fp表示错误的配对，tp表示成功的配对（成功的三元组会打上1），vis_tag会标记哪个gt_hois已经配对成功
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    true_hoi_details_for_one_image.append(hoi_detail)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                    false_hoi_details_for_one_image.append(hoi_detail)
                # score表示对应三元组的得分score
                self.score[triplet].append(pred_hoi['score'])
            self.true_hoi_details.append(true_hoi_details_for_one_image)
            self.false_hoi_details.append(false_hoi_details_for_one_image)

    # 计算预测的bbox列表与gts的bbox列表的iou
    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        # 超过标准的表示预测的bbox判定成功，否则判定为失败
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        # 对于pred的bbox列表中下标为i的bbox，match_pairs_dict[i]存储其对应的gts的bbox
        match_pairs_dict = {}
        # 对于pred的bbox列表中下标为i的bbox，match_pair_overlaps[i]存储其对应的gts的bbox与它的overlap
        match_pair_overlaps = {}
        # 这里如果有检测成功的bbox
        if iou_mat.max() > 0:
            # atch_pairs[1]是预测bbox的在列表中的下标
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0
    
    def judgePreds(self):
        # 对一张图片的pred和gts
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                # 如果gt_bboxes没有的话，直接所有预测的hoi全部判错
                hoi_details_for_one_image = []
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    hoi_detail = {}
                    hoi_detail['sub_box'] = pred_bboxes[pred_hoi['subject_id']]
                    hoi_detail['obj_box'] = pred_bboxes[pred_hoi['object_id']]
                    hoi_detail['category_id'] = pred_hoi['category_id']
                    hoi_detail['score'] = pred_hoi['score']
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    hoi_details_for_one_image.append(hoi_detail)
                    self.score[triplet].append(pred_hoi['score'])
                self.false_hoi_details.append(hoi_details_for_one_image)
                self.true_hoi_details.append([])

    def getPredsAndGts(self):
        return self.true_hoi_details, self.false_hoi_details, self.gts


# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import math
import functools
import numpy as np
import os

from pythia.common.sample import Sample

# ocr bbox processing 
def build_bbox_tensors(infos, max_length, feats, img_id, obj_bbox):

    # num of ocr bbox
    num_bbox = min(max_length, len(infos))
    # ocr bbox
    coord_tensor = torch.zeros((max_length, 4), dtype=torch.float)
    infos = infos[:num_bbox]
    sample = Sample()

    for idx, info in enumerate(infos):
        bbox = info["bounding_box"]
        if "top_left_x" in bbox:
            x = bbox["top_left_x"] # key might be 'topLeftX'
            y = bbox["top_left_y"] # key might be 'topLeftY'
        else:
            x = bbox["topLeftX"]
            y = bbox["topLeftY"]
        width = bbox["width"]
        height = bbox["height"]
        coord_tensor[idx][0] = x
        coord_tensor[idx][1] = y
        coord_tensor[idx][2] = x + width
        coord_tensor[idx][3] = y + height

    sample.coordinates = coord_tensor
    sample.ocr_mask = num_bbox

    image_path_org = './data/open_images/textvqa_gcy/'
    # image_path_org = './data/open_images/GT_OBJ_FRCN/'
    # image_path_org = './data/open_images/visual_genome/'


    oo_edge_path = image_path_org+'edge_oo/'
    ot_edge_path = image_path_org+'edge_ot/'
    tt_edge_path = image_path_org+'edge_tt/'
    to_edge_path = image_path_org+'edge_to/'

    set_name = search_file(image_path_org, img_id)
    knn_k = 5

    try:
        oo_node_matrix = torch.load(oo_edge_path+img_id + '_oo.pdh')
        sample.edge_oo = oo_node_matrix
        oo_feats = torch.load(oo_edge_path+img_id + '_oofeats.pdh')
        sample.edge_oofeats = oo_feats

        ot_node_matrix = torch.load(ot_edge_path+img_id + '_ot.pdh')
        sample.edge_ot = ot_node_matrix
        ot_feats = torch.load(ot_edge_path+img_id + '_otfeats.pdh')
        sample.edge_otfeats = ot_feats

        tt_node_matrix = torch.load(tt_edge_path+img_id + '_tt.pdh')
        sample.edge_tt = tt_node_matrix
        tt_feats = torch.load(tt_edge_path+img_id + '_ttfeats.pdh')
        sample.edge_ttfeats = tt_feats

        to_node_matrix = torch.load(to_edge_path+img_id + '_to.pdh')
        sample.edge_to = to_node_matrix
        to_feats = torch.load(to_edge_path+img_id + '_tofeats.pdh')
        sample.edge_tofeats = to_feats
    except:
        #Todo: generate obj-obj relation edge
        oo_node_matrix = finde_k_nearest_node(obj_bbox, knn_k)
        sample.edge_oo = oo_node_matrix
        oo_edge_file_name = oo_edge_path + img_id + "_oo.pdh"
        torch.save(oo_node_matrix, oo_edge_file_name)

        obj_obj_feat_variable = gen_oo_edge_feature(obj_bbox, oo_node_matrix, knn_k=knn_k)
        oo_edge_file_name = oo_edge_path + img_id + "_oofeats.pdh"
        torch.save(obj_obj_feat_variable, oo_edge_file_name)
        sample.edge_oofeats = obj_obj_feat_variable

        #Todo: generate object-text relation edge
        ot_node_matrix = dc_finde_k_nearest_node(obj_bbox, coord_tensor, knn_k)
        sample.edge_ot = ot_node_matrix
        ot_edge_file_name = ot_edge_path + img_id + "_ot.pdh"
        torch.save(ot_node_matrix, ot_edge_file_name)

        obj_text_feat_variable = gen_ot_edge_feature(obj_bbox, coord_tensor, ot_node_matrix, knn_k=knn_k)
        ot_edge_file_name = ot_edge_path + img_id + "_otfeats.pdh"
        torch.save(obj_text_feat_variable, ot_edge_file_name)
        sample.edge_otfeats = obj_text_feat_variable

        #Todo: generate text-text relation edge
        tt_node_matrix = finde_k_nearest_node(coord_tensor, knn_k)
        sample.edge_tt = tt_node_matrix
        tt_edge_file_name = tt_edge_path + img_id + "_tt.pdh"
        torch.save(tt_node_matrix, tt_edge_file_name)

        text_text_edge_feature = gen_tt_edge_feature(coord_tensor, tt_node_matrix, knn_k=knn_k) 
        tt_edge_file_name = tt_edge_path + img_id + "_ttfeats.pdh"
        torch.save(text_text_edge_feature, tt_edge_file_name)
        sample.edge_ttfeats = text_text_edge_feature

        #Todo: generate text-obj relation edge
        to_node_matrix = dc_finde_k_nearest_node(coord_tensor, obj_bbox, knn_k)
        sample.edge_to = to_node_matrix
        to_edge_file_name = to_edge_path + img_id + "_to.pdh"
        torch.save(to_node_matrix, to_edge_file_name)

        text_obj_feat_variable = gen_to_edge_feature(coord_tensor, obj_bbox, to_node_matrix, knn_k=knn_k)
        to_edge_file_name = to_edge_path + img_id + "_tofeats.pdh"
        torch.save(text_obj_feat_variable, to_edge_file_name)
        sample.edge_tofeats = text_obj_feat_variable
            
    return sample

def search_file(dir,sname): 
    train_dir = os.path.join(dir,'train/')
    files = os.listdir(train_dir)
    for file in files:
        if sname in file:
            return "train"
    test_dir = os.path.join(dir,'test/')
    files = os.listdir(test_dir)
    for file in files:
        if sname in file:
            return "test"

def finde_k_nearest_node(obj_bbox, knn_k=5):
    bbox_num, _ = obj_bbox.shape

    # node_matrix = np.ones((bbox_num, knn_k), dtype=np.int32)  # [bbox_num, k]
    node_matrix = torch.ones((bbox_num, knn_k)).long()
    node_matrix = node_matrix*-1
    for bbox_id in range(bbox_num):
        if(obj_bbox[bbox_id].sum() == 0):
            continue
        temp_sort_list = fetch_neighbour_ids(bbox_id, obj_bbox)
        node_matrix[bbox_id] = temp_sort_list[:knn_k]
    return node_matrix


def fetch_neighbour_ids(ref_id, bbox_list):
    """find neighborhood nodes
        For a given ref_ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not including itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
    """
    x1, y1, x2, y2 = bbox_list[ref_id]
    rx, ry = (x1 + x2) / 2, (y1 + y2) / 2

    def compare(bbox_id0, bbox_id1):
        x1, y1, x2, y2 = bbox_list[bbox_id0]
        ax0, ay0 = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = bbox_list[bbox_id1]
        ax1, ay1 = (x1 + x2) / 2, (y1 + y2) / 2
        # closer --> former
        if (rx - ax0) ** 2 + (ry - ay0) ** 2 <= (rx - ax1) ** 2 + (ry - ay1) ** 2:
            return -1
        else:
            return 1

    ann_ids = list(range(0, len(bbox_list)))  # copy in case the raw list is changed, sort the copy id list
    ann_ids = sorted(ann_ids, key=functools.cmp_to_key(compare))

    # neighborhood_node_id = np.ones(len(bbox_list), dtype=np.int32)
    neighborhood_node_id = torch.ones(len(bbox_list)).int()
    neighborhood_node_id = neighborhood_node_id*-1
    idx = 0
    for ann_id in ann_ids:
        if bbox_list[ref_id].sum() > 0 and ann_id != ref_id:
            if bbox_list[ann_id].sum() > 0:
                neighborhood_node_id[idx] = ann_id
            else:
                continue
            idx = idx+1

    return neighborhood_node_id

def dc_finde_k_nearest_node(ref_bbox, obj_bbox, knn_k=5):
    """
    find each ref_bbox's knn bbox in obj_bbox
    :param ref_bbox:
    :param obj_bbox:
    :param knn_k:
    :return:
    """
    bbox_num, _ = ref_bbox.shape

    node_matrix = torch.ones((bbox_num, knn_k)).long()
    node_matrix = node_matrix*-1
    # node_matrix = node_matrix*(obj_bbox.shape[0] - 1)
    for bbox_id in range(bbox_num):
        if(ref_bbox[bbox_id].sum() == 0):
            continue
        temp_sort_list = dc_fetch_neighbour_ids(bbox_id, ref_bbox, obj_bbox)
        node_matrix[bbox_id] = temp_sort_list[:knn_k]

    return node_matrix


def dc_fetch_neighbour_ids(ref_id, ref_bbox_list, bbox_list):
    """find neighborhood nodes
        For a given ref_ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not including itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
    """
    x1, y1, x2, y2 = ref_bbox_list[ref_id]
    rx, ry = (x1 + x2) / 2, (y1 + y2) / 2

    def compare(bbox_id0, bbox_id1):
        x1, y1, x2, y2 = bbox_list[bbox_id0]
        ax0, ay0 = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = bbox_list[bbox_id1]
        ax1, ay1 = (x1 + x2) / 2, (y1 + y2) / 2
        # closer --> former
        if (rx - ax0) ** 2 + (ry - ay0) ** 2 <= (rx - ax1) ** 2 + (ry - ay1) ** 2:
            return -1
        else:
            return 1

    ann_ids = list(range(0, len(bbox_list)))  # copy in case the raw list is changed, sort the copy id list
    ann_ids = sorted(ann_ids, key=functools.cmp_to_key(compare))

    neighborhood_node_id = torch.ones(len(bbox_list)).int()
    neighborhood_node_id = neighborhood_node_id*-1
    idx = 0
    for ann_id in ann_ids:
        if bbox_list[ann_id].sum() > 0:
            neighborhood_node_id[idx] = ann_id
        else:
            continue
        idx = idx+1

    return neighborhood_node_id

"""edge [location, y_feats]"""
def gen_oo_edge_feature(obj_bbox, k_nearest_node=None, knn_k=5):
        """
        generate enge feature for both obj-bbox and OCR-bbox
        process by sample
        generate each line and use torch.cat() to generate edge_loc_feats[]
        :param obj_bbox: (tlx, tly, brx, bry)
        :param image_size: (height, width)
        :return: edge features, only save k nearest edge (node_num, knn_node_num, 5+bbox_feats_dim)
        """
        bbox_num, _ = obj_bbox.shape
        edge_loc_feats = torch.zeros((bbox_num, knn_k, 5)).float()
        # temp = torch.zeros((bbox_num, bbox_num, 5)).float()
        temp_bbox = obj_bbox

        cx = (temp_bbox[:bbox_num, 0] + temp_bbox[:bbox_num, 2]) / 2
        cy = (temp_bbox[:bbox_num, 1] + temp_bbox[:bbox_num, 3]) / 2
        width = temp_bbox[:bbox_num, 2] - temp_bbox[:bbox_num, 0]
        height = temp_bbox[:bbox_num, 3] - temp_bbox[:bbox_num, 1]

        for i in range(bbox_num):
            if temp_bbox[i, :].sum() == 0:
                continue

            i_cx = cx[i]
            i_cy = cy[i]
            i_width = width[i]
            i_height = height[i]
            knn_temp = 0
            for pos_i in range(knn_k):
                j = k_nearest_node[i][pos_i]
                if j == -1 or temp_bbox[j, :].sum() == 0 or i_width== 0 or i_height == 0:
                    break
                j_tlx = temp_bbox[j][0]
                j_tly = temp_bbox[j][1]
                j_brx = temp_bbox[j][2]
                j_bry = temp_bbox[j][3]
                j_width = width[j]
                j_height = height[j]

                edge_loc_feats[i][knn_temp][0] = (j_tlx - i_cx)/i_width
                edge_loc_feats[i][knn_temp][1] = (j_tly - i_cy)/i_height
                edge_loc_feats[i][knn_temp][2] = (j_brx - i_cx)/i_width
                edge_loc_feats[i][knn_temp][3] = (j_bry - i_cy)/i_height
                edge_loc_feats[i][knn_temp][4] = (j_width*j_height)/(i_width*i_height)

                knn_temp = knn_temp+1

        return edge_loc_feats

def gen_tt_edge_feature(obj_bbox, k_nearest_node=None, knn_k=5):
    """
    generate enge feature for both obj-bbox and OCR-bbox
    process by sample
    generate each line and use torch.cat() to generate edge_loc_feats[]
    :param obj_bbox: (tlx, tly, brx, bry)
    :param image_size: (height, width)
    :return: edge features, only save k nearest edge (node_num, knn_node_num, 5+bbox_feats_dim)
    """
    bbox_num, _ = obj_bbox.shape

    edge_loc_feats = torch.zeros((bbox_num, knn_k, 5)).float()
    temp_bbox = obj_bbox

    cx = (temp_bbox[:bbox_num, 0] + temp_bbox[:bbox_num, 2]) / 2
    cy = (temp_bbox[:bbox_num, 1] + temp_bbox[:bbox_num, 3]) / 2
    width = temp_bbox[:bbox_num, 2] - temp_bbox[:bbox_num, 0]
    height = temp_bbox[:bbox_num, 3] - temp_bbox[:bbox_num, 1]

    for i in range(bbox_num):
        if temp_bbox[i, :].sum() == 0:
            continue

        i_cx = cx[i]
        i_cy = cy[i]
        i_width = width[i]
        i_height = height[i]
        knn_temp = 0
        for pos_i in range(knn_k):
            j = k_nearest_node[i][pos_i]
            if j == -1 or temp_bbox[j, :].sum() == 0 or i_width== 0 or i_height == 0:
                break
            j_tlx = temp_bbox[j][0]
            j_tly = temp_bbox[j][1]
            j_brx = temp_bbox[j][2]
            j_bry = temp_bbox[j][3]
            j_width = width[j]
            j_height = height[j]

            edge_loc_feats[i][knn_temp][0] = (j_tlx - i_cx) / i_width
            edge_loc_feats[i][knn_temp][1] = (j_tly - i_cy) / i_height
            edge_loc_feats[i][knn_temp][2] = (j_brx - i_cx) / i_width
            edge_loc_feats[i][knn_temp][3] = (j_bry - i_cy) / i_height
            edge_loc_feats[i][knn_temp][4] = (j_width * j_height) / (i_width * i_height)

            knn_temp = knn_temp + 1

    return edge_loc_feats

def gen_ot_edge_feature(obj_bbox, text_bbox, k_nearest_node=None, knn_k=5):
    """
    generate enge feature for object-text relation
    process by sample
    generate each line and use torch.cat() to generate edge_loc_feats[]
    :param obj_bbox: (tlx, tly, brx, bry), the bbox need to build knn edge
    :param text_bbox: (tlx, tly, brx, bry), the bbox of image_feat_variable
    :param image_size: (height, width)
    :return: edge features, only save k nearest edge (node_num, knn_node_num, 5+bbox_feats_dim)
    """
    bbox_num, _ = obj_bbox.shape
    edge_loc_feats = torch.zeros((bbox_num, knn_k, 5)).float()
    temp_bbox = obj_bbox

    cx = (temp_bbox[:bbox_num, 0] + temp_bbox[:bbox_num, 2]) / 2
    cy = (temp_bbox[:bbox_num, 1] + temp_bbox[:bbox_num, 3]) / 2
    width = temp_bbox[:bbox_num, 2] - temp_bbox[:bbox_num, 0]
    height = temp_bbox[:bbox_num, 3] - temp_bbox[:bbox_num, 1]

    for i in range(bbox_num):
        if temp_bbox[i, :].sum() == 0:
            continue

        i_cx = cx[i]
        i_cy = cy[i]
        i_width = width[i]
        i_height = height[i]
        knn_temp = 0
        for pos_i in range(knn_k):
            j = k_nearest_node[i][pos_i]
            if j == -1 or text_bbox[j, :].sum() == 0 or i_width== 0 or i_height == 0:
                break
            j_tlx = text_bbox[j][0]
            j_tly = text_bbox[j][1]
            j_brx = text_bbox[j][2]
            j_bry = text_bbox[j][3]
            j_width = j_brx - j_tlx
            j_height = j_bry - j_tly

            edge_loc_feats[i][knn_temp][0] = (j_tlx - i_cx) / i_width
            edge_loc_feats[i][knn_temp][1] = (j_tly - i_cy) / i_height
            edge_loc_feats[i][knn_temp][2] = (j_brx - i_cx) / i_width
            edge_loc_feats[i][knn_temp][3] = (j_bry - i_cy) / i_height
            edge_loc_feats[i][knn_temp][4] = (j_width * j_height) / (i_width * i_height)

            knn_temp = knn_temp + 1

    return edge_loc_feats

def gen_to_edge_feature(obj_bbox, text_bbox, k_nearest_node=None, knn_k=5):
    """
    generate enge feature for object-text relation
    process by sample
    generate each line and use torch.cat() to generate edge_loc_feats[]
    :param obj_bbox: (tlx, tly, brx, bry), the bbox need to build knn edge
    :param text_bbox: (tlx, tly, brx, bry), the bbox of image_feat_variable
    :param image_size: (height, width)
    :return: edge features, only save k nearest edge (node_num, knn_node_num, 5+bbox_feats_dim)
    """
    bbox_num, _ = obj_bbox.shape

    edge_loc_feats = torch.zeros((bbox_num, knn_k, 5)).float()
    temp_bbox = obj_bbox

    cx = (temp_bbox[:bbox_num, 0] + temp_bbox[:bbox_num, 2]) / 2
    cy = (temp_bbox[:bbox_num, 1] + temp_bbox[:bbox_num, 3]) / 2
    width = temp_bbox[:bbox_num, 2] - temp_bbox[:bbox_num, 0]
    height = temp_bbox[:bbox_num, 3] - temp_bbox[:bbox_num, 1]

    for i in range(bbox_num):
        if temp_bbox[i, :].sum() == 0:
            continue

        i_cx = cx[i]
        i_cy = cy[i]
        i_width = width[i]
        i_height = height[i]
        knn_temp = 0
        for pos_i in range(knn_k):
            j = k_nearest_node[i][pos_i]
            if j == -1 or text_bbox[j, :].sum() == 0 or i_width== 0 or i_height == 0:
                break
            j_tlx = text_bbox[j][0]
            j_tly = text_bbox[j][1]
            j_brx = text_bbox[j][2]
            j_bry = text_bbox[j][3]
            j_width = j_brx - j_tlx
            j_height = j_bry - j_tly

            edge_loc_feats[i][knn_temp][0] = (j_tlx - i_cx) / i_width
            edge_loc_feats[i][knn_temp][1] = (j_tly - i_cy) / i_height
            edge_loc_feats[i][knn_temp][2] = (j_brx - i_cx) / i_width
            edge_loc_feats[i][knn_temp][3] = (j_bry - i_cy) / i_height
            edge_loc_feats[i][knn_temp][4] = (j_width * j_height) / (i_width * i_height)

            knn_temp = knn_temp + 1

    return edge_loc_feats
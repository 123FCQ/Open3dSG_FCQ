#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This source code is from ScanNet
#   (https://github.com/ScanNet/ScanNet)
# Copyright (c) 2017 ScanNet authors
# This source code is licensed under the ScanNet license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

"""
Build Instance level input of ScanNet
"""
import json
import os
import numpy as np
import trimesh
from open3dsg.config import define
from open3dsg import util


def scannet_get_instance_ply(plydata, segs, aggre, random_color=False):
    ''' map idx to segments '''
    seg_map = dict()
    for idx in range(len(segs['segIndices'])):
        seg = segs['segIndices'][idx]
        if seg in seg_map:
            seg_map[seg].append(idx)
        else:
            seg_map[seg] = [idx]

    ''' Group segments '''
    aggre_seg_map = dict()
    for segGroup in aggre['segGroups']:
        aggre_seg_map[segGroup['id']] = list()
        for seg in segGroup['segments']:
            aggre_seg_map[segGroup['id']].extend(seg_map[seg])
    assert (len(aggre_seg_map) == len(aggre['segGroups']))
    # print('num of aggre_seg_map:',len(aggre_seg_map))

    ''' Generate random colors '''
    if random_color:
        colormap = dict()
        for seg in aggre_seg_map.keys():
            colormap[seg] = util.color_rgb(util.rand_24_bit())

    ''' Over write label to segments'''
    # labels = plydata.vertices[:,0] # wrong but work around
    try:
        labels = plydata.metadata['_ply_raw']['vertex']['data']['label']
    except:
        labels = plydata.elements[0]['label']

    instances = np.zeros_like(labels)
    colors = plydata.visual.vertex_colors
    used_vts = set()
    for seg, indices in aggre_seg_map.items():
        s = set(indices)
        if len(used_vts.intersection(s)) > 0:
            raise RuntimeError('duplicate vertex')
        used_vts.union(s)
        for idx in indices:
            instances[idx] = seg
            if random_color:
                colors[idx][0] = colormap[seg][0]
                colors[idx][1] = colormap[seg][1]
                colors[idx][2] = colormap[seg][2]
    return plydata, instances


def load_scannet(pth_ply, pth_agg, pth_seg, verbose=False, random_color=False):
    ''' Load GT '''
    plydata = trimesh.load(pth_ply, process=False) # pth_ply: 比如scene0191_00_vh_clean_2.labels.ply 点云数据
    num_verts = plydata.vertices.shape[0]  # 顶点的个数 
    if verbose:
        print('num of verts:', num_verts)

    ''' Load segment file'''
    with open(pth_seg) as f: # pth_seg: 比如scene0191_00_vh_clean_2.0.010000.segs.json 分割数据 
        segs = json.load(f)
    if verbose:
        print('len(aggre[\'segIndices\']):', len(segs['segIndices']))
    segment_ids = list(np.unique(np.array(segs['segIndices'])))  # 分割ID 每个分割ID包含多个顶点
    if verbose:
        print('num of unique ids:', len(segment_ids))

    ''' Load aggregation file'''
    with open(pth_agg) as f: # pth_agg: 比如scene0191_00.aggregation.json 聚合数据
        aggre = json.load(f)
    # assert(aggre['sceneId'].split('scannet.')[1]==scan_id)
    # assert(aggre['segmentsFile'].split('scannet.')[1] == scan_id+args.segs)

    plydata, instances = scannet_get_instance_ply(plydata, segs, aggre, random_color=random_color)
    # instances 每个顶点的实例ID (145736,) 指的是聚合文件中的segGroup的id

    labels = plydata.metadata['_ply_raw']['vertex']['data']['label'].flatten() # (145736,) 点云数据中原本携带的
    points = plydata.vertices # (145736, 3) 每一行为顶点的xyz坐标

    # the label is in the range of 1 to 40. 0 is unlabeled
    # instance 0 is unlabeled.
    # labels vs instances 
    # labels 存储的是语义分割信息，它表示每个点所属的物体类别,例如：所有椅子的点都会被标记为相同的类别ID（如例如ID 7），即使是不同的椅子
    # instances 存储的是实例分割信息，它表示每个点所属的具体物体实例，例如：房间中的两把不同椅子会有不同的实例ID（如实例ID 3和实例ID 8）
    return plydata, points, labels, instances


if __name__ == '__main__':

    # read split file
    print('read split file..')
    ids = open(define.SCANNET_SPLIT_VAL).read().splitlines()
    print('there are', len(ids), 'sequences')

    for scan_id in sorted(ids):
        # scan_id = 'scene0000_00'
        print('scan_id', scan_id)

        ''' Load point cloud '''
        plydata = trimesh.load(os.path.join(define.SCANNET_DATA_PATH, scan_id, scan_id+define.SCANNET_PLY_SUBFIX), process=False)
        num_verts = plydata.vertices.shape[0]
        print('num of verts:', num_verts)

        ''' Load segment file'''
        with open(os.path.join(define.SCANNET_DATA_PATH, scan_id, scan_id+define.SCANNET_SEG_SUBFIX)) as f:
            segs = json.load(f)
        assert (segs['sceneId'] == scan_id)
        print('len(aggre[\'segIndices\']):', len(segs['segIndices']))
        tmp = np.array(segs['segIndices'])
        segment_ids = list(np.unique(tmp))  # get unique segment ids
        print('num of unique ids:', len(segment_ids))

        ''' Load aggregation file'''
        with open(os.path.join(define.SCANNET_DATA_PATH, scan_id, scan_id+define.SCANNET_AGGRE_SUBFIX)) as f:
            aggre = json.load(f)
        assert (aggre['sceneId'].split('scannet.')[1] == scan_id)
        assert (aggre['segmentsFile'].split('scannet.')[1] == scan_id+define.SCANNET_SEG_SUBFIX)

        plydata = scannet_get_instance_ply(plydata, segs, aggre)
        plydata[0].export('tmp_segments.ply')

        break

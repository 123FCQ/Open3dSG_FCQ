#!/usr/bin/env python3
"""
3RScan场景分析检查工具
用于诊断场景预处理失败的原因，不修改原始代码
"""

import os
import json
import pickle
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.distance import cdist
import argparse
import sys

# 如果需要导入原始配置，取消下面的注释并修改路径
# sys.path.append('/path/to/open3dsg')
# from open3dsg.config.config import CONF

class SceneAnalyzer:
    def __init__(self, r3scan_raw_path, r3scan_processed_path):
        """
        初始化场景分析器
        
        Args:
            r3scan_raw_path: 3RScan原始数据路径
            r3scan_processed_path: 3RScan处理后数据路径
        """
        self.r3scan_raw_path = r3scan_raw_path
        self.r3scan_processed_path = r3scan_processed_path
        
        # 加载类别和关系映射
        self.word2idx = self._load_categories()
        self.rel2idx = self._load_relationships()
        
        # 加载包围盒数据
        self.boxes_train = self._load_json_safe(os.path.join(r3scan_raw_path, 'obj_boxes_train_refined.json'))
        self.boxes_val = self._load_json_safe(os.path.join(r3scan_raw_path, 'obj_boxes_val_refined.json'))
    
    def _load_categories(self):
        """加载类别映射"""
        word2idx = {}
        classes_file = os.path.join(self.r3scan_raw_path, "3DSSG_subset/classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                for idx, line in enumerate(f):
                    category = line.strip()
                    if category:
                        word2idx[category] = idx
        return word2idx
    
    def _load_relationships(self):
        """加载关系映射"""
        rel2idx = {}
        rel_file = os.path.join(self.r3scan_raw_path, "3DSSG_subset/relationships.txt")
        if os.path.exists(rel_file):
            with open(rel_file, 'r') as f:
                for idx, line in enumerate(f):
                    relationship = line.strip()
                    if relationship:
                        rel2idx[relationship] = idx
        return rel2idx
    
    def _load_json_safe(self, filepath):
        """安全加载JSON文件"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def load_mesh(self, mesh_path, texture_path):
        """加载网格和纹理"""
        try:
            mesh = trimesh.load(mesh_path, process=False)
            im = Image.open(texture_path)
            tex = trimesh.visual.TextureVisuals(image=im)
            mesh.visual.texture = tex
            return mesh.vertices, mesh.visual.to_color().vertex_colors[:, :3], mesh.vertex_normals
        except Exception as e:
            print(f"加载网格失败: {e}")
            return None, None, None
    
    def within_bbox2(self, p, obb):
        """检查点是否在有向包围盒内"""
        p = p[:, :3]
        center = np.array(obb["centroid"])
        axis_len = np.array(obb["axesLengths"])
        axis_x = np.array(obb["normalizedAxes"][0:3])
        axis_y = np.array(obb["normalizedAxes"][3:6])
        axis_z = np.array(obb["normalizedAxes"][6:9])
        project_x = np.sum((axis_x[None]*(p - center)), axis=1)
        project_y = np.sum((axis_y[None]*(p - center)), axis=1)
        project_z = np.sum((axis_z[None]*(p - center)), axis=1)
        return (-axis_len[0]/2 <= project_x) & (project_x <= axis_len[0]/2) & \
               (-axis_len[1]/2 <= project_y) & (project_y <= axis_len[1]/2) & \
               (-axis_len[2]/2 <= project_z) & (project_z <= axis_len[2]/2)
    
    def analyze_scene_files(self, scan_id):
        """分析场景文件完整性"""
        print(f"=== 场景文件完整性分析: {scan_id} ===")
        
        scan = scan_id[:-2]
        required_files = {
            'mesh': f"{scan}/mesh.refined.v2.obj",
            'texture': f"{scan}/mesh.refined_0.png",
            'segments': f"{scan}/mesh.refined.0.010000.segs.v2.json",
            'semantic': f"{scan}/semseg.v2.json"
        }
        
        file_status = {}
        for name, file_path in required_files.items():
            full_path = os.path.join(self.r3scan_raw_path, file_path)
            exists = os.path.exists(full_path)
            file_status[name] = exists
            print(f"  {'✓' if exists else '✗'} {name}: {file_path}")
            
        # 检查object2image文件 - 尝试多个可能的路径
        possible_object2image_paths = [
            # 原始路径格式
            os.path.join(self.r3scan_processed_path, 'views', f"{scan_id}_object2image.pkl"),
            # 不带split的路径格式
            os.path.join(self.r3scan_processed_path, 'views', f"{scan}_object2image.pkl"),
            # output路径格式
            os.path.join("open3dsg/output/datasets/OpenSG_3RScan/views", f"{scan}_object2image.pkl"),
            # 相对于processed路径的output格式
            os.path.join(self.r3scan_processed_path, "../output/datasets/OpenSG_3RScan/views", f"{scan}_object2image.pkl"),
        ]
        
        object2image_path = None
        for path in possible_object2image_paths:
            if os.path.exists(path):
                object2image_path = path
                file_status['object2image'] = True
                print(f"  ✓ object2image: {path}")
                break
        
        if object2image_path is None:
            file_status['object2image'] = False
            print(f"  ✗ object2image: 未找到 (尝试了以下路径:)")
            for path in possible_object2image_paths:
                print(f"      - {path}")
        
        # 存储找到的路径供后续使用
        self.object2image_path = object2image_path
        
        return file_status
    
    def analyze_relationships(self, scan_id):
        """分析关系数据"""
        print(f"\n=== 关系数据分析: {scan_id} ===")
        
        # 查找关系数据
        relationships_files = [
            os.path.join(self.r3scan_raw_path, "3DSSG_subset/relationships_train.json"),
            os.path.join(self.r3scan_raw_path, "3DSSG_subset/relationships_validation.json")
        ]
        
        scan_relationships = None
        source_file = None
        
        for rel_file in relationships_files:
            if os.path.exists(rel_file):
                with open(rel_file, 'r') as f:
                    data = json.load(f)
                    for scan_data in data["scans"]:
                        if scan_data["scan"] + "-" + str(hex(scan_data["split"]))[-1] == scan_id:
                            scan_relationships = scan_data
                            source_file = rel_file
                            break
                if scan_relationships:
                    break
        
        if not scan_relationships:
            print("  ❌ 未找到对应的关系数据")
            return None
            
        print(f"  ✓ 找到关系数据，来源: {os.path.basename(source_file)}")
        print(f"  ✓ 对象数量: {len(scan_relationships['objects'])}")
        print(f"  ✓ 关系数量: {len(scan_relationships['relationships'])}")
        
        # 分析对象类别分布
        object_categories = {}
        for obj_id, obj_info in scan_relationships['objects'].items():
            # obj_info 可能是字符串（直接是类别名）或字典
            if isinstance(obj_info, str):
                category = obj_info
            elif isinstance(obj_info, dict):
                category = obj_info.get('label', 'unknown')
            else:
                category = 'unknown'
            object_categories[category] = object_categories.get(category, 0) + 1
        
        print(f"  ✓ 对象类别分布: {dict(sorted(object_categories.items()))}")
        
        return scan_relationships
    
    def analyze_segmentation(self, scan_id):
        """分析分割数据"""
        print(f"\n=== 分割数据分析: {scan_id} ===")
        
        scan = scan_id[:-2]
        
        # 分析点分割
        seg_file = os.path.join(self.r3scan_raw_path, f"{scan}/mesh.refined.0.010000.segs.v2.json")
        if not os.path.exists(seg_file):
            print("  ❌ 分割文件不存在")
            return None, None
            
        with open(seg_file, 'r') as f:
            seg_data = json.load(f)
            seg_indices = seg_data["segIndices"]
        
        print(f"  ✓ 分割点数量: {len(seg_indices)}")
        unique_segs, seg_counts = np.unique(seg_indices, return_counts=True)
        print(f"  ✓ 唯一分割ID数量: {len(unique_segs)}")
        print(f"  ✓ 分割大小统计: min={min(seg_counts)}, max={max(seg_counts)}, mean={np.mean(seg_counts):.1f}")
        
        # 分析语义分割
        semseg_file = os.path.join(self.r3scan_raw_path, f"{scan}/semseg.v2.json")
        if not os.path.exists(semseg_file):
            print("  ❌ 语义分割文件不存在")
            return seg_indices, None
            
        with open(semseg_file, 'r') as f:
            semseg_data = json.load(f)
            seg_groups = semseg_data["segGroups"]
        
        print(f"  ✓ 对象组数量: {len(seg_groups)}")
        
        # 分析对象类别
        categories = {}
        object_sizes = []
        for obj in seg_groups:
            label = obj.get("label", "unknown")
            categories[label] = categories.get(label, 0) + 1
            object_sizes.append(len(obj.get("segments", [])))
        
        print(f"  ✓ 语义类别分布: {dict(sorted(categories.items()))}")
        print(f"  ✓ 对象大小统计: min={min(object_sizes)}, max={max(object_sizes)}, mean={np.mean(object_sizes):.1f}")
        
        return seg_indices, seg_groups
    
    def analyze_object_visibility(self, scan_id, relationships_scan):
        """分析对象可见性"""
        print(f"\n=== 对象可见性分析: {scan_id} ===")
        
        if not hasattr(self, 'object2image_path') or self.object2image_path is None:
            print("  ❌ 对象-图像映射文件不存在")
            return {}
            
        try:
            with open(self.object2image_path, "rb") as f:
                object2fame_all = pickle.load(f)
        except Exception as e:
            print(f"  ❌ 读取对象-图像映射文件失败: {e}")
            return {}
        
        print(f"  ✓ 成功加载对象-图像映射文件: {self.object2image_path}")
        print(f"  ✓ 文件中包含的对象总数: {len(object2fame_all)}")
        
        # 如果是合并文件，需要根据scan_id筛选相关对象
        scan = scan_id[:-2]
        split = scan_id[-1]
        
        # 检查文件结构
        sample_keys = list(object2fame_all.keys())[:5]
        print(f"  ✓ 示例对象ID: {sample_keys}")
        
        # 根据关系文件中的对象ID筛选
        relationship_obj_ids = set(int(k) for k in relationships_scan["objects"].keys())
        print(f"  ✓ 关系文件中的对象ID: {sorted(relationship_obj_ids)}")
        
        # 筛选出当前split相关的对象映射
        object2fame = {}
        for obj_id in relationship_obj_ids:
            obj_key = str(obj_id)
            if obj_key in object2fame_all:
                object2fame[obj_key] = object2fame_all[obj_key]
        
        print(f"  ✓ 当前split相关的映射对象数量: {len(object2fame)}")
        
        # 检查关系中的对象有多少有图像映射
        mapped_objects = 0
        visible_objects = []
        invisible_objects = []
        
        for obj_id in relationship_obj_ids:
            if object2fame.get(str(obj_id), None) is not None:
                mapped_objects += 1
                visible_objects.append(obj_id)
                frames_info = object2fame[str(obj_id)]
                print(f"    对象 {obj_id}: {len(frames_info)} 个可见帧")
            else:
                invisible_objects.append(obj_id)
        
        print(f"  ✓ 有图像映射的对象数量: {mapped_objects}/{len(relationship_obj_ids)}")
        print(f"  ✓ 映射覆盖率: {mapped_objects/len(relationship_obj_ids)*100:.1f}%")
        
        if invisible_objects:
            print(f"  ⚠️  不可见对象: {invisible_objects}")
        
        if mapped_objects < 4:
            print("  ❌ 有图像映射的对象数量少于4个，预处理可能失败")
        
        return object2fame
    
    def analyze_point_cloud_processing(self, scan_id, relationships_scan, seg_indices, seg_groups):
        """分析点云处理过程"""
        print(f"\n=== 点云处理分析: {scan_id} ===")
        
        scan = scan_id[:-2]
        
        # 加载网格数据
        mesh_path = os.path.join(self.r3scan_raw_path, f"{scan}/mesh.refined.v2.obj")
        texture_path = os.path.join(self.r3scan_raw_path, f"{scan}/mesh.refined_0.png")
        
        pcl_array, rgb_array, normals_array = self.load_mesh(mesh_path, texture_path)
        if pcl_array is None:
            print("  ❌ 无法加载点云数据")
            return None
            
        print(f"  ✓ 点云总数: {len(pcl_array)}")
        pcl_array = np.concatenate((pcl_array, rgb_array, normals_array), axis=1)
        
        # 构建分割到对象的映射
        seg2obj = {}
        objects = {}
        obb = {}
        labels = {}
        obj_id_list = [int(k) for k in relationships_scan["objects"].keys()]
        
        valid_objects = 0
        for o in seg_groups:
            obj_id = o["id"]
            if obj_id not in obj_id_list:
                continue
            if o["label"] not in self.word2idx:
                continue
                
            valid_objects += 1
            labels[obj_id] = o["label"]
            segs = o["segments"]
            objects[obj_id] = []
            obb[obj_id] = o["obb"]
            
            for seg_id in segs:
                seg2obj[seg_id] = obj_id
        
        print(f"  ✓ 有效对象数量: {valid_objects}")
        
        # 分析每个对象的点云
        object_point_counts = {}
        for obj_id in objects.keys():
            point_count = 0
            for index, seg_id in enumerate(seg_indices):
                if index >= len(pcl_array):
                    continue
                if seg_id in seg2obj and seg2obj[seg_id] == obj_id:
                    point_count += 1
            object_point_counts[obj_id] = point_count
        
        print(f"  ✓ 对象点云统计:")
        for obj_id, count in sorted(object_point_counts.items()):
            print(f"    对象 {obj_id} ({labels.get(obj_id, 'unknown')}): {count} 点")
        
        empty_objects = [obj_id for obj_id, count in object_point_counts.items() if count == 0]
        if empty_objects:
            print(f"  ⚠️  空对象: {empty_objects}")
        
        return pcl_array, seg_indices, seg2obj, obb, objects, labels
    
    def analyze_relationship_processing(self, scan_id, relationships_scan, pcl_array, seg_indices, seg2obj, obb):
        """分析关系处理过程"""
        print(f"\n=== 关系处理分析: {scan_id} ===")
        
        # 添加数据结构调试信息
        print(f"  调试信息:")
        print(f"    点云数组形状: {pcl_array.shape}")
        print(f"    分割索引长度: {len(seg_indices)}")
        print(f"    分割到对象映射数量: {len(seg2obj)}")
        print(f"    包围盒数量: {len(obb)}")
        
        # 构建关系对
        triples = []
        pairs = []
        relationships_triples = relationships_scan["relationships"]
        objects_id = list(obb.keys())
        
        for triple in relationships_triples:
            if (triple[0] not in objects_id) or (triple[1] not in objects_id) or (triple[0] == triple[1]):
                continue
            triples.append(triple[:3])
            if triple[:2] not in pairs:
                pairs.append(triple[:2])
        
        # 补充'none'关系
        for i in objects_id:
            for j in objects_id:
                if i == j or [i, j] in pairs:
                    continue
                triples.append([i, j, 0])
                pairs.append(([i, j]))
        
        print(f"  ✓ 总关系对数量: {len(pairs)}")
        print(f"  ✓ 有标签关系数量: {len([t for t in triples if t[2] != 0])}")
        
        # 分析前几个关系对的联合包围盒
        problematic_pairs = []
        successful_pairs = []
        
        for i, pair in enumerate(pairs[:10]):  # 只分析前10对
            s, o = pair
            print(f"\n  关系对 {i}: 主体={s}, 客体={o}")
            
            if s not in obb or o not in obb:
                print(f"    ❌ 包围盒信息缺失")
                problematic_pairs.append((i, pair, "missing_obb"))
                continue
            
            # 分析联合包围盒中的点
            bbox_mask = self.within_bbox2(pcl_array, obb[s]) | self.within_bbox2(pcl_array, obb[o])
            points_in_union = np.sum(bbox_mask)
            print(f"    联合包围盒中的点数: {points_in_union}")
            
            if points_in_union == 0:
                print(f"    ❌ 联合包围盒中没有点")
                problematic_pairs.append((i, pair, "empty_union_bbox"))
                continue
            
            # 确保索引数组长度匹配
            min_length = min(len(seg_indices), len(bbox_mask), len(pcl_array))
            bbox_mask = bbox_mask[:min_length]
            seg_indices_subset = seg_indices[:min_length]
            
            # 分析这些点的归属
            union_seg_indices = np.array(seg_indices_subset)[bbox_mask]
            s_points = 0
            o_points = 0
            other_points = 0
            
            for seg_id in union_seg_indices:
                if seg_id in seg2obj:
                    if seg2obj[seg_id] == s:
                        s_points += 1
                    elif seg2obj[seg_id] == o:
                        o_points += 1
                    else:
                        other_points += 1
                else:
                    other_points += 1
            
            print(f"    主体对象点数: {s_points}")
            print(f"    客体对象点数: {o_points}")
            print(f"    其他对象点数: {other_points}")
            
            if s_points == 0:
                print(f"    ❌ 主体对象在联合包围盒中没有点")
                problematic_pairs.append((i, pair, "empty_subject"))
                continue
                
            if o_points == 0:
                print(f"    ❌ 客体对象在联合包围盒中没有点")
                problematic_pairs.append((i, pair, "empty_object"))
                continue
            
            # 模拟距离计算
            try:
                # 构建点云标记
                union_pcl = []
                for index, point in enumerate(pcl_array):
                    if index >= min_length:
                        break
                    if bbox_mask[index]:
                        if index < len(seg_indices_subset) and seg_indices_subset[index] in seg2obj:
                            if seg2obj[seg_indices_subset[index]] == s:
                                point_labeled = np.append(point, 1)
                            elif seg2obj[seg_indices_subset[index]] == o:
                                point_labeled = np.append(point, 2)
                            else:
                                point_labeled = np.append(point, 0)
                            union_pcl.append(point_labeled)
                
                if len(union_pcl) > 0:
                    union_pcl = np.array(union_pcl)
                    s_cloud = union_pcl[union_pcl[:, -1] == 1]
                    o_cloud = union_pcl[union_pcl[:, -1] == 2]
                    
                    if len(s_cloud) > 0 and len(o_cloud) > 0:
                        # 尝试计算距离
                        c_s = np.mean(s_cloud[:, :3], axis=0)
                        c_o = np.mean(o_cloud[:, :3], axis=0)
                        center_dist = np.linalg.norm(c_s - c_o)
                        
                        mag_dist = cdist(s_cloud[:, :3], o_cloud[:, :3])
                        min_dist = np.min(mag_dist)
                        
                        print(f"    ✓ 中心距离: {center_dist:.3f}")
                        print(f"    ✓ 最小距离: {min_dist:.3f}")
                        successful_pairs.append((i, pair))
                    else:
                        print(f"    ❌ 标记后的点云为空")
                        problematic_pairs.append((i, pair, "empty_labeled_cloud"))
                else:
                    print(f"    ❌ 联合点云为空")
                    problematic_pairs.append((i, pair, "empty_union_cloud"))
                    
            except Exception as e:
                print(f"    ❌ 距离计算失败: {e}")
                problematic_pairs.append((i, pair, f"distance_error: {e}"))
        
        print(f"\n  成功处理的关系对: {len(successful_pairs)}")
        print(f"  有问题的关系对: {len(problematic_pairs)}")
        
        if problematic_pairs:
            print(f"  问题类型统计:")
            problem_types = {}
            for _, _, problem_type in problematic_pairs:
                problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
            for problem_type, count in problem_types.items():
                print(f"    {problem_type}: {count}")
        
        return successful_pairs, problematic_pairs
    
    def run_full_analysis(self, scan_id):
        """运行完整分析"""
        print(f"开始分析场景: {scan_id}")
        print("=" * 60)
        
        # 1. 文件完整性检查
        file_status = self.analyze_scene_files(scan_id)
        if not all(file_status.values()):
            print("\n❌ 文件完整性检查失败，无法继续分析")
            return
        
        # 2. 关系数据分析
        relationships_scan = self.analyze_relationships(scan_id)
        if not relationships_scan:
            print("\n❌ 关系数据分析失败，无法继续分析")
            return
        
        # 3. 分割数据分析
        seg_indices, seg_groups = self.analyze_segmentation(scan_id)
        if seg_indices is None or seg_groups is None:
            print("\n❌ 分割数据分析失败，无法继续分析")
            return
        
        # 4. 对象可见性分析
        object2fame = self.analyze_object_visibility(scan_id, relationships_scan)
        if len(object2fame) == 0:
            print("\n❌ 对象可见性分析失败，无法继续分析")
            return
        
        # 5. 点云处理分析
        result = self.analyze_point_cloud_processing(scan_id, relationships_scan, seg_indices, seg_groups)
        if result is None:
            print("\n❌ 点云处理分析失败，无法继续分析")
            return
        
        pcl_array, seg_indices, seg2obj, obb, objects, labels = result
        
        # 6. 关系处理分析
        successful_pairs, problematic_pairs = self.analyze_relationship_processing(
            scan_id, relationships_scan, pcl_array, seg_indices, seg2obj, obb
        )
        
        # 7. 生成总结报告
        print(f"\n=== 分析总结: {scan_id} ===")
        print(f"✓ 文件完整性: 通过")
        print(f"✓ 关系数据: {len(relationships_scan['objects'])} 对象, {len(relationships_scan['relationships'])} 关系")
        print(f"✓ 分割数据: {len(seg_indices)} 点, {len(seg_groups)} 对象组")
        print(f"✓ 对象可见性: {len(object2fame)} 可见对象")
        print(f"✓ 点云处理: {len(pcl_array)} 点, {len(objects)} 有效对象")
        print(f"✓ 关系处理: {len(successful_pairs)} 成功, {len(problematic_pairs)} 失败")
        
        if len(successful_pairs) == 0:
            print(f"\n❌ 预测结果: 该场景预处理会失败 - 没有成功的关系对")
        elif len(successful_pairs) < len(successful_pairs) + len(problematic_pairs) * 0.5:
            print(f"\n⚠️  预测结果: 该场景预处理可能不稳定 - 成功率较低")
        else:
            print(f"\n✓ 预测结果: 该场景预处理应该能够成功")


def main():
    print("场景分析工具启动...")
    
    parser = argparse.ArgumentParser(description='3RScan场景分析工具')
    parser.add_argument('--scan_id', type=str, required=True, help='场景ID，如 scene0000_00-0')
    parser.add_argument('--r3scan_raw', type=str, required=True, help='3RScan原始数据路径')
    parser.add_argument('--r3scan_processed', type=str, required=True, help='3RScan处理后数据路径')
    
    print("解析命令行参数...")
    args = parser.parse_args()
    
    print(f"参数: scan_id={args.scan_id}")
    print(f"参数: r3scan_raw={args.r3scan_raw}")
    print(f"参数: r3scan_processed={args.r3scan_processed}")
    
    print("初始化分析器...")
    try:
        analyzer = SceneAnalyzer(args.r3scan_raw, args.r3scan_processed)
        print("开始分析...")
        analyzer.run_full_analysis(args.scan_id)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
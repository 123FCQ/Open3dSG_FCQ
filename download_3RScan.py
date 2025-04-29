#!/usr/bin/env python
# 下载3RScan公共数据集

# 数据基于Creative Commons Attribution-NonCommercial-ShareAlike 4.0协议发布
# 有用信息：每个扫描都有一个唯一ID，列表位于：
# http://campar.in.tum.de/public_datasets/3RScan/scans.txt 或 https://github.com/WaldJohannaU/3RScan/tree/master/splits

# 脚本用法：
# - 下载整个3RScan数据集：download.py -o [下载目录]
# - 下载特定扫描(例如 19eda6f4-55aa-29a0-8893-8eac3a4d8193)：download.py -o [下载目录] --id 19eda6f4-55aa-29a0-8893-8eac3a4d8193
# - 下载tfrecords文件：download.py -o [下载目录] --type=tfrecords
# - 相关的元数据文件位于：http://campar.in.tum.de/public_datasets/3RScan/3RScan.json
# - 3RScan的3D语义场景图可在项目页面下载：https://3dssg.github.io

# 导入必要的系统模块
import sys
import argparse
import os
from tqdm import tqdm
import requests
# 根据Python版本选择合适的urllib模块
if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib
import tempfile
import re

# 定义基础URL和数据URL常量
BASE_URL = 'http://campar.in.tum.de/public_datasets/3RScan/'
DATA_URL = BASE_URL + 'Dataset/'
TOS_URL = 'http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf'

# 定义测试文件类型列表
TEST_FILETYPES = ['mesh.refined.v2.obj', 'mesh.refined.mtl', 'mesh.refined_0.png', 'sequence.zip']

# 定义所有文件类型列表，包括测试文件类型和额外的语义标注文件
# 注意：语义标注仅提供给训练集、验证集和测试集中的参考扫描
FILETYPES = TEST_FILETYPES + ['labels.instances.annotated.v2.ply', 'mesh.refined.0.010000.segs.v2.json', 'semseg.v2.json']

# 定义发布版本和隐藏测试集的文件名
RELEASE = 'release_scans.txt'
HIDDEN_RELEASE = 'test_rescans.txt'

# 定义发布版本大小和ID正则表达式模式
RELEASE_SIZE = '~94GB'
id_reg = re.compile(r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}")

class DownloadProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        downloaded = block_num * block_size
        self.pbar.update(block_size)
        
        if downloaded >= total_size:
            self.pbar.close()

def get_scans(scan_file):
    """从给定的URL获取扫描ID列表"""
    scan_lines = urllib.urlopen(scan_file)
    scans = []
    for scan_line in scan_lines:
        # 解码每行并去除换行符
        scan_line = scan_line.decode('utf8').rstrip('\n')
        # 使用正则表达式匹配ID
        match = id_reg.search(scan_line)
        if match:
            scan_id = match.group()
            scans.append(scan_id)
    return scans

def download_release(release_scans, out_dir, file_types):
    """下载整个发布版本的数据"""
    print('Downloading 3RScan release to ' + out_dir + '...')
    for scan_id in tqdm(release_scans, desc="Overall progress"):
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types)
    # for scan_id in release_scans:
    #     scan_out_dir = os.path.join(out_dir, scan_id)
    #     download_scan(scan_id, scan_out_dir, file_types)
    print('Downloaded 3RScan release.')

def download_file(url, out_file):
    """下载单个文件"""
    print(url)
    
    # 创建输出目录（如果不存在）
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # 如果文件不存在则下载
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        
        # 创建临时文件
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        os.close(fh)  # 立即关闭文件句柄
        
        try:
            # 下载文件到临时位置
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 检查请求是否成功
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 使用二进制模式写入临时文件
            with open(out_file_tmp, 'wb') as f, tqdm(
                desc=os.path.basename(out_file),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            # 重命名为最终文件名
            os.rename(out_file_tmp, out_file)
            
        except Exception as e:
            print(f"下载出错: {e}")
            # 清理临时文件
            if os.path.exists(out_file_tmp):
                os.remove(out_file_tmp)
            raise  # 重新抛出异常
            
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types):
    """下载单个扫描的所有相关文件"""
    print('Downloading 3RScan scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # 下载每种文件类型
    for ft in file_types:
        url = DATA_URL + '/' + scan_id + '/' + ft
        out_file = out_dir + '/' + ft
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)

def download_tfrecord(url, out_dir, file):
    """下载TFRecord文件"""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, file)
    download_file(url + '/' + file, out_file)

def main():
    """主函数：处理命令行参数并执行相应的下载任务"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Downloads 3RScan public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--type', help='specific file type to download')
    args = parser.parse_args()

    # 显示使用条款提示
    print('By pressing any key to continue you confirm that you have agreed to the 3RScan terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')

    # 获取发布版本和测试版本的扫描ID
    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)
    file_types = FILETYPES
    file_types_test = TEST_FILETYPES

    if args.type:  # 下载特定类型文件
        file_type = args.type
        if file_type == 'tfrecords':
            # 下载TFRecord文件
            download_tfrecord(BASE_URL, args.out_dir, 'val-scans.tfrecords')
            download_tfrecord(BASE_URL, args.out_dir, 'train-scans.tfrecords')
            return
        elif file_type not in FILETYPES:
            print('ERROR: Invalid file type: ' + file_type)
            return
        # 设置要下载的文件类型
        file_types = [file_type]
        if file_type not in TEST_FILETYPES:
            file_types_test = []
        else:
            file_types_test = [file_type]
            
    if args.id:  # 下载单个扫描
        scan_id = args.id
        if scan_id not in release_scans and scan_id not in test_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            out_dir = os.path.join(args.out_dir, scan_id)
            # 根据扫描ID所属集合选择下载方式
            if scan_id in release_scans:
                download_scan(scan_id, out_dir, file_types)
            elif scan_id in test_scans:
                download_scan(scan_id, out_dir, file_types_test)
    else: # 下载整个数据集
        # 显示警告信息
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire 3RScan release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all 3RScan scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        # 下载所有文件
        download_release(release_scans, args.out_dir, file_types)
        download_release(test_scans, args.out_dir, file_types_test)

# 程序入口点
if __name__ == "__main__": main()
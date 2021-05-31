import pandas as pd
import os
import pickle as pkl
import click
from pathlib import Path
import re
from lxml import etree
import numpy as np
import pdb


@click.group()
def main():
    pass


def parse_attribute_xml(xml_path):
    attri_tree = etree.parse(str(xml_path))
    tree_root = attri_tree.getroot()
    item_list = tree_root.find('Items').findall('Item')
    attri_map = {}
    for item in item_list:
        item_attri = item.attrib
        attri_map[item_attri['imageName']] = [int(item_attri['colorID']) -1 , int(item_attri['typeID']) -1]
    return attri_map

def parse_key_view_point_txt(txt_path):  
    def _parse_key_point(content_list):
        # key_point_location = np.array(content_list[1:-1]).astype(np.int).reshape(-1, 2)
        # assert key_point_location.shape[0] == 20, key_point_location.shape[0]
        # key_point_index = np.where(np.sum(key_point_location, axis=1) != -2)[0]
        # return key_point_index

        key_point_location = np.array(content_list[1:-1]).astype(np.int)
        assert key_point_location.size == 40, key_point_location.size
        return key_point_location

    img2kp_view = {}

    with open(str(txt_path)) as f:
        for line_content in f.readlines():
            content_list = line_content.strip().split()
            img_name = content_list[0].split('/')[-1]
            key_point_index = _parse_key_point(content_list)
            view_label = int(content_list[-1])
            img2kp_view[img_name] = (key_point_index, view_label)
    return img2kp_view

def parse_veri776_meta(input_path):
    """解析veri776数据集的元数据"""
    train_attri_xml = input_path / 'train_label.xml'
    test_attri_xml = input_path / 'test_label.xml'
    train_kp_txt = input_path / 'VehicleReIDKeyPointData-master' / 'keypoint_train.txt'
    test_kp_txt = input_path / 'VehicleReIDKeyPointData-master' / 'keypoint_test.txt'

    train_attri = parse_attribute_xml(train_attri_xml)
    train_kp = parse_key_view_point_txt(train_kp_txt)
    train_img_with_attri_kp = set(train_attri.keys()) & set(train_kp.keys())

    test_attri = parse_attribute_xml(test_attri_xml)
    test_kp = parse_key_view_point_txt(test_kp_txt)
    test_img_with_attri_kp = set(test_attri.keys()) & set(test_kp.keys())

    meta_data = {}
    meta_data['train'] = [train_attri, train_kp]
    meta_data['query'] = [test_attri, test_kp]
    meta_data['gallery'] = [test_attri, test_kp]

    chosen_img = {}
    chosen_img['train'] = train_img_with_attri_kp
    chosen_img['query'] = test_img_with_attri_kp
    chosen_img['gallery'] = test_img_with_attri_kp

    return meta_data, chosen_img

veri776_mask_dir = '/home/nfs/em5/reid_group/private/qifengliang/PVEN/examples/outputs/veri776_masks'
@main.command()
@click.option("--input-path", required=True)
@click.option("--output-path", default="veri776.pkl")
def veri776(input_path, output_path):
    input_path = os.path.abspath(input_path)
    output_dir = os.path.split(output_path)[0]
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_path).absolute()

    meta_data, chosen_img = parse_veri776_meta(input_path)
    # pdb.set_trace()

    output_dict = {}

    pattern = re.compile(r"(\d+)_c(\d+)_.+\.jpg")
    for phase in ["train", "query", "gallery"]:
        output_dict[phase] = []
        sub_path = input_path / f"image_{phase}"
        if phase == "gallery":
            sub_path = input_path / f"image_test"
        raw_nums = 0   # 用于统计原有split中的样本数量
        final_nums = 0 # 用于统计那些含有属性，角度，关键点标注的样本数量
        for image_path in sub_path.iterdir():
            raw_nums += 1
            sample = {}
            image_name = image_path.name
            if image_name not in chosen_img[phase]:
                print("No included {}".format(image_path))
                continue
            final_nums += 1
            v_id, camera = pattern.match(image_name).groups()
            sample["filename"] = image_name
            sample["image_path"] = str(image_path)
            sample["id"] = v_id
            sample["cam"] = camera
            sample["color"] = meta_data[phase][0][image_name][0]
            sample["type"] = meta_data[phase][0][image_name][1]
            sample["kp"] = meta_data[phase][1][image_name][0]
            sample["view"] = meta_data[phase][1][image_name][1]
            sample["mask_path"] = os.path.join(veri776_mask_dir, phase, os.path.splitext(str(image_name))[0] + '.png')
            output_dict[phase].append(sample)
        print('There are {} samples in raw {} split, and {} samples are used'.format(raw_nums, phase, final_nums))
    with open(output_path, "wb") as f:
        pkl.dump(output_dict, f)

    


@main.command()
@click.option('--input-path', default='/data1/dechao_meng/mengdechao/datasets/VehicleID_V1.0')
@click.option('--output-path', default='../outputs/vehicleid.pkl')
def vehicleid(input_path, output_path):
    input_path = os.path.abspath(input_path)
    PATH = input_path

    images = {}

    images['train']        = open(PATH + '/train_test_split/train_list.txt').read().strip().split('\n')
    images['gallery_800']   = open(PATH + '/train_test_split/test_list_800.txt').read().strip().split('\n')
    images['gallery_1600']   = open(PATH + '/train_test_split/test_list_1600.txt').read().strip().split('\n')
    images['gallery_2400']   = open(PATH + '/train_test_split/test_list_2400.txt').read().strip().split('\n')
    images['query_800']   = []
    images['query_1600']   = []
    images['query_2400']   = []

    outputs = {}
    for key, lists in images.items():
        output = []
        for img_name in lists:
            item = {
                "image_path": f"{PATH}/image/{img_name.split(' ')[0]}.jpg",
                "name": img_name,
                "id": img_name.split(' ')[1],
                "cam": 0
            }
            output.append(item)
        outputs[key] = output
        
    base_path = os.path.split(output_path)[0]
    if base_path != '' and not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    with open(output_path, 'wb') as f:
        pkl.dump(outputs, f)

@main.command()
@click.option('--input-path', default='/home/aa/mengdechao/datasets/veriwild')
@click.option('--output-path', default='../outputs/veriwild.pkl')
def veriwild(input_path, output_path):
    input_path = os.path.abspath(input_path)
    PATH = input_path

    images = {}

    images['train']        = open(PATH + '/train_test_split/train_list.txt').read().strip().split('\n')
    images['query_3000']   = open(PATH + '/train_test_split/test_3000_query.txt').read().strip().split('\n')
    images['gallery_3000'] = open(PATH + '/train_test_split/test_3000.txt').read().strip().split('\n')
    images['query_5000']   = open(PATH + '/train_test_split/test_5000_query.txt').read().strip().split('\n')
    images['gallery_5000'] = open(PATH + '/train_test_split/test_5000.txt').read().strip().split('\n')
    images['query_10000']  = open(PATH + '/train_test_split/test_10000_query.txt').read().strip().split('\n')
    images['gallery_10000']= open(PATH + '/train_test_split/test_10000.txt').read().strip().split('\n')

    wild_df = pd.read_csv(f'{PATH}/train_test_split/vehicle_info.txt', sep=';', index_col='id/image')

    # Pandas indexing is very slow, change it to dict
    wild_dict = wild_df.to_dict()
    camid_dict = wild_dict['Camera ID']

    outputs = {}
    for key, lists in images.items():
        output = []
        for img_name in lists:
            item = {
                "image_path": f"{PATH}/images/{img_name}.jpg",
                "name": img_name,
                "id": img_name.split('/')[0],
    #             "cam": wild_df.loc[img_name]['Camera ID'] 
                "cam": camid_dict[img_name]
            }
            output.append(item)
        outputs[key] = output
        
    base_path = os.path.split(output_path)[0]
    if base_path != '' and not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    with open(output_path, 'wb') as f:
        pkl.dump(outputs, f)


if __name__ == "__main__":
    main()
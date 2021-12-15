from pycocotools.coco import COCO
import pandas as pd
import os
from shutil import copyfile

def make_nested_dir(path):
    dirs = path.split('/')
    curr = '.'
    for dir_ in dirs:
        curr = os.path.join(curr, dir_)
        if not os.path.isdir(curr):
            os.mkdir(curr)

# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # basic parameters of an image
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        # retrieve metadata for all persons in the current image
        anns = coco.loadAnns(ann_ids)

        yield [img_id, img_file_name, w, h, anns]

def convert_to_df(coco):
    images_data = []
    persons_data = []
    # iterate over all images
    for img_id, img_fname, w, h, meta in get_meta(coco):
        images_data.append({
            'image_id': int(img_id),
            'path': img_fname,
            'width': int(w),
            'height': int(h)
        })
        # iterate over all metadata
        for m in meta:
            persons_data.append({
                'image_id': m['image_id'],
                'is_crowd': m['iscrowd'],
                'bbox': m['bbox'],
                'area': m['area'],
                'num_keypoints': m['num_keypoints'],
                'keypoints': m['keypoints'],
            })
    # create dataframe with image paths
    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)
    # create dataframe with persons
    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('image_id', inplace=True)
    return images_df, persons_df

if __name__ == '__main__':
    train_annot_path = 'data/annotations/person_keypoints_train2017.json'
    val_annot_path = 'data/annotations/person_keypoints_val2017.json'
    train_images_path_src = 'data/images/train2017'
    val_images_path_src = 'data/images/val2017'
    train_images_path_dest = 'data/keypoints_images/train2017'
    val_images_path_dest = 'data/keypoints_images/val2017'

    print('Creating COCO indices')
    train_coco = COCO(train_annot_path)
    val_coco = COCO(val_annot_path)

    print('Parsing files')
    images_df, persons_df = convert_to_df(train_coco, 'train')
    train_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    train_coco_df = train_coco_df[train_coco_df.num_keypoints>0]
    images_df, persons_df = convert_to_df(val_coco, 'val')
    val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    val_coco_df = val_coco_df[val_coco_df.num_keypoints>0]
    train_idx = train_coco_df.path.tolist()
    val_idx = val_coco_df.path.tolist()

    print('Creating destination directories')
    make_nested_dir(train_images_path_dest)
    make_nested_dir(val_images_path_dest)

    print('Copying files')
    for fn in train_idx:
        src = os.path.join(train_images_path_src, fn)
        dest = os.path.join(train_images_path_dest, fn)
        copyfile(src, dest)
    for fn in val_idx:
        src = os.path.join(val_images_path_src, fn)
        dest = os.path.join(val_images_path_dest, fn)
        copyfile(src, dest)
    
    print('Done')

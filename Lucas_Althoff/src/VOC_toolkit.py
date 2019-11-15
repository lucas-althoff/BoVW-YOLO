import os
import pandas as pd
from bs4 import BeautifulSoup
import voc_utils
from more_itertools import unique_everseen

root_dir = r'C:\Users\usuario\Desktop\VOC2007_DataSet\VOCdevkit\VOC2007'
#root_dir = 'C:/Users/usuario/Desktop/VOC2007_DataSet/VOCdevkit/VOC2007/Annotations'
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

image_sets = ['aeroplane', 'bicycle', 'bird', 'boat', 
'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
'sofa', 'train', 'trainval', 'tvmonitor', 'val']

# category name is from above, dataset is either "train" or
# "val" or "train_val"
def imgs_from_category(cat_name, dataset):
    filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
    df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['filename', 'true'])
    return df

def imgs_from_category_as_list(cat_name, dataset):
    df = imgs_from_category(cat_name, dataset)
    df = df[df['true'] == 1]
    return df['filename'].values

def annotation_file_from_img(img_name):
    return os.path.join(ann_dir, str(img_name)) + '.xml'

# annotation operations
def load_annotation(img_filename):
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)

def get_all_obj_and_box(objname, img_set):
    img_list = imgs_from_category_as_list(objname, img_set)
    
    for img in img_list:
        annotation = load_annotation(img)

# image operations
def load_img(img_filename):
    return io.load_image(os.path.join(img_dir, img_filename + '.jpg'))

train_img_list = imgs_from_category_as_list('bicycle', 'train')
train_img_list2 = []
### ADD: adjust list of str to xml annotation naming
for temp in train_img_list:
    temp ='0'*(6 - len(str(temp))) + str(temp) 
    train_img_list2.append(temp)
a = load_annotation(train_img_list2[0])

def load_train_data(category):
    to_find = category
    train_filename = r'C:\Users\usuario\Desktop\VOC2007_DataSet\VOCdevkit\VOC2007\csvs\train_' + category + '.csv'
    if os.path.isfile(train_filename):
        return pd.read_csv(train_filename)
    else:
        train_img_list = imgs_from_category_as_list(to_find, 'train')
        data = []
        for item in train_img_list:
            ### ADD: adjust list of str to xml annotation naming
            item ='0'*(6 - len(str(item))) + str(item) 
            anno = load_annotation(item)
            objs = anno.findAll('object')
            for obj in objs:
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    if str(name_tag.contents[0]) == 'bicycle':
                        fname = anno.findChild('filename').contents[0]
                        bbox = obj.findChildren('bndbox')[0]
                        xmin = int(bbox.findChildren('xmin')[0].contents[0])
                        ymin = int(bbox.findChildren('ymin')[0].contents[0])
                        xmax = int(bbox.findChildren('xmax')[0].contents[0])
                        ymax = int(bbox.findChildren('ymax')[0].contents[0])
                        data.append([fname, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax'])
        df.to_csv(train_filename)
        return df

df = load_train_data('bicycle')
print(list(unique_everseen(list(voc_utils.img_dir + df['fname']))))

for cat in image_sets:
    if cat != 'train' and cat != 'val' and cat != 'trainval':
        load_train_data(cat)

df = load_train_data('person')
for row_num, entry in df.iterrows():
    print(entry)
print(df)
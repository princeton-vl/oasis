from torch.utils import data
from os.path import join, splitext
import os
import cv2
import random
import numpy as np
import pickle


def save_obj(obj, name, verbal=False ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if verbal:
            print(" Done saving %s" % name)


def load_obj(name, verbal=False):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
        if verbal:
            print(" Done loading %s" % name)
        return obj


def vis_mask(mask, filename):
    a = mask.astype(np.float32)
    a = a - np.min(a)
    a = a / np.max(a)
    a = a * 255.0

    cv2.imwrite(filename, a.astype(np.uint8))


class data_augmenter():
    def __init__(self, width, height):
        """
            Args:
                width and height are only used to determine the
                output aspect ratio, not the actual output size
        """
        self.ops = []
        cv2.setNumThreads(0)
        self.width = float(width)
        self.height = float(height)

    def add_rotation(self, probability, max_left_rotation=-10, max_right_rotation=10):
        self.ops.append({'type':'rotation', 'probability':probability, 'max_left_rotation': max_left_rotation, 'max_right_rotation':max_right_rotation})
    def add_zoom(self, probability, min_percentage, max_percentage):
        self.ops.append({'type':'zoom', 'probability':probability, 'min_percentage': min_percentage, 'max_percentage': max_percentage})
    def add_flip_left_right(self, probability):
        self.ops.append({'type':'flip_lr', 'probability':probability})
    def add_crop(self, probability, min_percentage=0.5):
        self.ops.append({'type':'crop', 'probability':probability, 'min_percentage':min_percentage})

    def __str__(self):
        out_str = 'Data Augmenter:\n'
        for op in self.ops:
            out_str += '\t'
            for key in op.keys():
                out_str = out_str + str(key) +':'+ str(op[key]) + '\t'
            out_str += '\n'
        return out_str

    def aug(self, img, ROI, label):
        orig_img = img.copy()
        orig_ROI = ROI.copy()
        orig_lab = label.copy()

        for op in self.ops:
            if random.uniform(0.0, 1.0) <= op['probability']:
                if op['type'] == 'crop':
                    percentage = random.uniform(op['min_percentage'], 1.0)

                    #################### image
                    if img.shape[0] <= img.shape[1]:
                        dst_h = int(img.shape[0] * percentage)
                        dst_w = min(int(dst_h / self.height * self.width), img.shape[1])
                    elif img.shape[0] > img.shape[1]:
                        dst_w = int(img.shape[1] * percentage)
                        dst_h = min(int(dst_w / self.width * self.height), img.shape[0])
                    offset_y = random.randint(0, img.shape[0]- dst_h)
                    offset_x = random.randint(0, img.shape[1]- dst_w)

                    img = img[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w, :]
                    ROI = ROI[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w]
                    label = label[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w, :]

                    if np.sum(ROI) == 0:
                        # print("Fail at cropping")
                        img = orig_img.copy()
                        ROI = orig_ROI.copy()
                        label = orig_lab.copy()

                elif op['type'] == 'flip_lr':
                    # print "Flipping..................."
                    #################### image
                    img = cv2.flip(img, 1)
                    ROI = cv2.flip(ROI, 1)
                    label = cv2.flip(label, 1)

                elif op['type'] == 'zoom':
                    # print "Zooming..................."
                    #################### image
                    percentage = random.uniform(op['min_percentage'], op['max_percentage'])
                    img = cv2.resize(img, dsize=None, fx = percentage, fy = percentage)
                    ROI = cv2.resize(ROI, dsize=None, fx = percentage, fy = percentage, interpolation = cv2.INTER_NEAREST)
                    label = cv2.resize(label, dsize=None, fx = percentage, fy = percentage)

                elif op['type'] == 'rotation':
                    # print "Rotating..................."
                    #################### image
                    angle = random.uniform(-op['max_left_rotation'], op['max_right_rotation'])
                    rotation_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
                    ROI = cv2.warpAffine(ROI, rotation_matrix, (ROI.shape[1], ROI.shape[0]))
                    label = cv2.warpAffine(label, rotation_matrix, (label.shape[1], label.shape[0]))

                    if np.sum(ROI) == 0:
                        # print("Fail at rotation")
                        img = orig_img.copy()
                        ROI = orig_ROI.copy()
                        label = orig_lab.copy()

                    # self.draw(img, target, '3_rotation.png')

        return img, ROI, label


class OASISDataset(data.Dataset):
    def __init__(self, dataset_dir='./data/OASIS_occ_fold_trainval', split='train', width = 640, height = 480, is_data_aug=False):
        print("####################################")
        print("Using OASIS dataset")
        print("split = %s" % split)
        print("dataset_dir = %s" % dataset_dir)
        print("height = %d, width = %d" % (width, height))

        # Set dataset directory and split.
        self.dataset_dir = dataset_dir
        self.split       = split
        ann_file = {'train': os.path.join(dataset_dir,'train.txt'),
            'val': os.path.join(dataset_dir,'val.txt'),
            'test': os.path.join(dataset_dir,'test.txt'),
            'train_25': os.path.join(dataset_dir,'train_25.txt'),
            'train_50': os.path.join(dataset_dir,'train_50.txt'),
        }[split]

        self.width = width
        self.height = height

        self.parse_ann_file(ann_file)


        self.da = data_augmenter(width = self.width, height = self.height)
        if is_data_aug:
       	    self.da.add_zoom(probability = 0.8, min_percentage = 0.4, max_percentage = 2.0)
            self.da.add_crop(probability = 1.0, min_percentage = 0.5)
            self.da.add_rotation(probability = 0.8, max_left_rotation = -10.0, max_right_rotation = 10.0)
            self.da.add_flip_left_right(probability = 0.5)

        print(self.da)
        print("# of samples = %d" % self.n_sample)
        print("####################################")

    def parse_ann_file(self, ann_file):
        # ann_file: a txt file, where each line is in this form:
        #   path_to_img, path_to_label_img
        lines = [line.strip() for line in open(ann_file, 'r')]
        self.images_name    = []
        self.list_of_images = []
        self.list_of_labels = []
        self.list_of_ROI    = []

        dataset_meta_file = ann_file.replace(".txt", "_meta.pkl")
        if not os.path.exists(dataset_meta_file):
            for line in lines:
                # This implementation makes the number of output channels adaptoble.
                items = line.split(",")
                for i, _ in enumerate(items):
                    items[i] = items[i].strip()
                img_fname, ROI_fname = items[0], items[1]
                label_fnames = items[2:]
                
                # valid_instance = True

                # if self.split == 'train':
                #     valid_instance = valid_instance and os.path.exists(os.path.join(self.dataset_dir, img_fname))
                #     valid_instance = valid_instance and os.path.exists(os.path.join(self.dataset_dir, ROI_fname))
                #     for label_fname in label_fnames:
                #         valid_instance = valid_instance and os.path.exists(os.path.join(self.dataset_dir, label_fname))

                # if valid_instance:
                
                name, ext = os.path.splitext(os.path.basename(img_fname))
                self.images_name.append(name)
                self.list_of_images.append(img_fname)
                self.list_of_ROI.append(ROI_fname)
                self.list_of_labels.append(label_fnames)

                # else:
                #     print("%s does not exist." % (img_fname))

            save_obj({'images_name'   :self.images_name,
                      'list_of_images':self.list_of_images,
                      'list_of_labels':self.list_of_labels,
                      'list_of_ROI'   :self.list_of_ROI},
                      dataset_meta_file)

        else:
            _temp = load_obj(dataset_meta_file)
            self.images_name    = _temp['images_name']
            self.list_of_images = _temp['list_of_images']
            self.list_of_labels = _temp['list_of_labels']
            self.list_of_ROI    = _temp['list_of_ROI']

        self.n_sample = len(self.list_of_images)


    def __len__(self):
        return self.n_sample


    def __getitem__(self, index):
        edge = None

        if self.split == "train":
            # Get edge.
            edges = []
            for label in self.list_of_labels[index]:
                edge_path = join(self.dataset_dir, label)
                geo_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
                geo_edge = geo_edge > 0
                geo_edge = geo_edge.astype(np.float32)
                edges.append(geo_edge)
            # Append null labels
            null_edge = (sum(edges) == 0)
            null_edge = null_edge.astype(np.float32)
            edges.append(null_edge)
            edge = np.stack(edges, axis=2)

            # Get ROI.
            ROI_path = os.path.join(self.dataset_dir, self.list_of_ROI[index])
            ROI = cv2.imread(ROI_path)
            if len(ROI.shape) == 3:
                ROI = ROI[:,:,0]
            ROI = ROI > 0
            ROI = ROI.astype(np.float32)

        # Get image.
        image_path = join(self.dataset_dir, self.list_of_images[index])
        image = cv2.imread(image_path).astype(np.float32)

        # Note: Image arrays read by OpenCV and Matplotlib are slightly different.
        # Matplotlib reading code:
        #   image = plt.imread(image_path).astype(np.float32)
        #   image = image[:, :, ::-1]            # RGB to BGR.
        # Reference:
        #   https://oldpan.me/archives/python-opencv-pil-dif
        image = image - np.array((104.00698793,  # Minus statistics.
                                  116.66876762,
                                  122.67891434))
        image = image.astype(np.float32)         # To float32.


        # Return image and (possible) edge.
        if self.split.find('train') >= 0:
            h, w, c = image.shape
            # resize to a fixed resolution
            image, ROI, edge = self.da.aug(image, ROI, edge)

            image = cv2.resize(image, (self.width, self.height))
            ROI = cv2.resize(ROI, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
            edge = cv2.resize(edge, (self.width, self.height))

            ROI = ROI > 0
            ROI = ROI.astype(np.float32)
            edge = edge > 0.2 # Labels are always hard
            edge = edge.astype(np.float32)

            #transform
            image = np.transpose(image, (2, 0, 1))   # HWC to CHW.
            edge = np.transpose(edge, (2, 0, 1))   # HWC to CHW.
            ROI  = ROI[np.newaxis, :, :]  # Add one channel at first (CHW).
            return image, edge, ROI, h, w
        else:
            h, w, c = image.shape
            # print(h, w)
            image = cv2.resize(image, (self.width, self.height))
            image = np.transpose(image, (2, 0, 1))   # HWC to CHW.
            image = image.astype(np.float32)         # To float32.

            edge = np.zeros((self.height, self.width, 3))
            edge = np.transpose(edge, (2, 0, 1))   # HWC to CHW.
            edge = edge.astype(np.float32)
            return image, edge, np.zeros(edge.shape), h, w

import os
import cv2
import pickle
import numpy as np

def load_obj(name, verbal=False):
    with open(name, 'rb') as f:
        try:
            obj = pickle.load(f)
        except:
            obj = pickle.load(f, encoding='bytes')
        if verbal:
            print( " Done loading %s" % name)
        return obj


def create_meta_data(csv_filename):
    image_list = []
    
    with open(csv_filename, "r") as f:
        lines = [line.strip() for line in f]
        lines = lines[1:]   #skip the header
    
    '''
        Image,FocalLength,Mask,Normal,Depth,RelativeDepth,Fold,Occlusion,SharpOcc,SmoothOcc,SurfaceSemantic,PlanarInstance,SmoothInstance,ContinuousInstance
    '''
    for line in lines:
        splits = line.split(",")
        
        img_name = splits[0]
        focal = float(splits[1])
        depth = splits[4]
        image_list.append((img_name, focal, depth))

    return image_list


def create_mesh_files(sample_ID, img_path, XYZ, output_folder):
    height, width = XYZ.shape[:2]
    XYZ_to_idx = {}
    idx = 1
    with open("%s/%s.mtl" % (output_folder, sample_ID), "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 0.200000 0.200000 0.200000\n")
        f.write("Kd 0.752941 0.752941 0.752941\n")
        f.write("Ks 1.000000 1.000000 1.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 2\n")
        f.write("Ns 0.000000\n")
        f.write("map_Ka %s\n" % img_path)
        f.write("map_Kd %s\n" % img_path)

    with open("%s/%s.obj" % (output_folder, sample_ID), "w") as f:
        f.write("mtllib %s.mtl\n" % sample_ID)
        for y in range(height):
            for x in range(width):
                if XYZ[y][x][2] > 0:

                    XYZ_to_idx[(y, x)] = idx
                    idx += 1

                    f.write("v %.4f %.4f %.4f\n" % (XYZ[y][x][0], XYZ[y][x][1], XYZ[y][x][2]))
                    f.write("vt %.8f %.8f\n" % ( float(x) / float(width), 1.0 - float(y) / float(height)))
        
        f.write("usemtl material_0\n")

        for y in range(height-1):
            for x in range(width-1):
                if XYZ[y][x][2] > 0 and XYZ[y][x+1][2] > 0 and XYZ[y+1][x][2] > 0:
                    f.write("f %d/%d %d/%d %d/%d\n" % (XYZ_to_idx[(y, x)], XYZ_to_idx[(y, x)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y+1, x)], XYZ_to_idx[(y+1, x)]))
                if XYZ[y][x+1][2] > 0 and XYZ[y+1][x+1][2] > 0 and XYZ[y+1][x][2] > 0:
                    f.write("f %d/%d %d/%d %d/%d\n" % (XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y+1, x+1)], XYZ_to_idx[(y+1, x+1)], XYZ_to_idx[(y+1, x)], XYZ_to_idx[(y+1, x)]))
            

def unpack_depth(depth_dict, resolution):
    out_depth = np.zeros(resolution, dtype = np.float32)
    out_depth.fill(-1.0)
    
    min_x = depth_dict['min_x']
    max_x = depth_dict['max_x']
    min_y = depth_dict['min_y']
    max_y = depth_dict['max_y']
    roi_depth = depth_dict['depth']        
    roi_depth[np.isnan(roi_depth)] = -1.0		
    out_depth[min_y:max_y+1, min_x:max_x+1] = roi_depth

    return out_depth
    
def back_project(depth, f):
    height, width  = depth.shape[:2]
    
    xs = np.linspace(0, width-1, width)
    ys = np.linspace(0, height-1, height)
    xv, yv = np.meshgrid(xs, ys)
    
    xv -= 0.5 * width
    yv -= 0.5 * height


    X = np.multiply(xv / (f + 1e-8), depth)
    Y = np.multiply(yv / (f + 1e-8), depth)
    Z = depth

    XYZ = np.stack([X,Y,Z], axis = -1)

    return XYZ



if __name__ == "__main__":
    
    csv_file = "OASIS/OASIS_trainval/OASIS_val.csv"
    output_folder = "OASIS/OASIS_trainval/vis_mesh"

    img_list = create_meta_data(csv_file)
    os.makedirs(output_folder, exist_ok=True)

    for img_name, focal, depth_f in img_list:
        img = cv2.imread(img_name)
        depth = unpack_depth(load_obj(depth_f), img.shape[:2])
        XYZ = back_project(depth, focal)

        sample_ID = os.path.basename(img_name).replace(".png", "")

        create_mesh_files(sample_ID, img_name, XYZ, output_folder = output_folder)
        
        
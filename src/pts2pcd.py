import open3d as o3d
import numpy as np
import sys
import os 
import re

def extract_numbers_from_filename(filename):
    """
    Extract numbers from a filename in the format "a-b-c.txt".
    
    Args:
    - filename (str): The filename.
    
    Returns:
    - tuple of int: Tuple containing the extracted numbers.
    """
    filename_only = filename.split('/')[-1]
    pattern = r'(-?\d+)_(-?\d+)_(-?\d+)\.txt'
    match = re.match(pattern, filename_only)
    if match:
        numbers = tuple(map(int, match.groups()))
        return numbers
    else:
        return None

def get_files_in_folder(folder_path):
    """
    Get the paths of all files in a folder.
    
    Args:
    - folder_path (str): The path to the folder.
    
    Returns:
    - list of str: List containing paths of all files in the folder.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    return file_paths

def read_point_cloud(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            points.append([x, y, z])
    return np.array(points, dtype=np.float32)

def visualize_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualization
    o3d.visualization.draw_geometries([pcd])

def save_point_cloud_pcd(point_cloud, output_file, downsampled = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    o3d.io.write_point_cloud(output_file, pcd)
    
    if downsampled:
        output_pcd_file = output_pcd_file[:-4]+'_downsampled.pcd'
        pcd = pcd.voxel_down_sample(voxel_size=0.2)
        o3d.io.write_point_cloud(output_file, pcd)
    

if __name__ == "__main__":
    # file_path = "data/1-139--8--44.txt"
    # output_pcd_file = "data/1-139--8--44.pcd"
    file_name = "131--6--36"
    file_name = "test_01"

    folder_base = "data/simulation_scan/"
    folder_path = folder_base + "txt/"
    file_paths = get_files_in_folder(folder_path)
    print("Files in folder:")


    folder_pt_path = folder_base + "pcd/"
    try:
        os.makedirs(folder_pt_path)        
    except OSError:
        print(f"Creation of the directory {folder_pt_path} failed")

    for file_path in file_paths:
        print(extract_numbers_from_filename(file_path))
                
        filename_only = file_path.split('/')[-1][:-4]
        output_pcd_file = folder_pt_path + filename_only+".pcd"
        print(output_pcd_file)
        print("read points...", file_path)
        point_cloud = read_point_cloud(file_path)

        downsampled = False
        # if len(sys.argv) == 2:
        #     downsampled = True
        #     output_pcd_file = output_pcd_file[:-4]+'_downsampled.pcd'
        print("saving...")
        save_point_cloud_pcd(point_cloud, output_pcd_file, downsampled=downsampled)
        # print("visualizing...")
        # visualize_point_cloud(point_cloud)


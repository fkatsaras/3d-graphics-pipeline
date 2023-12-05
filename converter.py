import numpy as np
import point_cloud_utils as pcu
import tqdm


def mesh_to_pc(input_files, output_folder, n_points):
    
    
    '''
    Function to convert a set of mesh files (.obj) from ShapeNet into a set of point cloud (.npy) files via
    Lloyd's algorithm (https://en.wikipedia.org/wiki/Lloyd%27s_algorithm). This simply wraps code from 
    Python Point Cloud Utils (PCU) by Francis Williams to allow for easier utilization with folders:
    https://github.com/fwilliams/point-cloud-utils
    
    Args:
        input_files (str): List of .obj file paths
        output_folder (str): Folder path where the resulting .npy files should be stored
        
    Returns:
        bool: True when file conversion is completed
    
    '''

    counter = 0
    
    for file in tqdm.tqdm(input_files, total=len(input_files)):
        # v is a nv by 3 NumPy array of vertices
        # f is an nf by 3 NumPy array of face indexes into v 
        # n is a nv by 3 NumPy array of vertex normals
        try:
            v, f, n = pcu.read_obj(file)
        except ValueError:
            print("Could not read: " + str(file))
            pass

        pc = pcu.sample_mesh_lloyd(v, f, n_points)
        output_file = output_folder + str(counter) + '.npy'
        np.save(output_file, pc)
        counter += 1


inputfiles = ["C:\\Users\\user\\Desktop\\AUTh\\8TH SEMESTER\\ΓΡΑΦΙΚΗ ΜΕ ΥΠΟΛΟΓΙΣΤΕΣ\\ΕΡΓΑΣΙΕΣ\\3d-graphics-pipeline\\model.obj"]
outputfiles = "C:\\Users\\user\\Desktop\\AUTh\\8TH SEMESTER\\ΓΡΑΦΙΚΗ ΜΕ ΥΠΟΛΟΓΙΣΤΕΣ\\ΕΡΓΑΣΙΕΣ\\3d-graphics-pipeline"

result = mesh_to_pc(input_files= inputfiles, output_folder= outputfiles, n_points=1000)


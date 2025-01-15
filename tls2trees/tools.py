import glob
import itertools
import os
import shutil
import string
import threading

import laspy
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
import sys



def read_ply(fp, newline=None):
    if (sys.version_info > (3, 0)):
        open_file = open(fp, encoding='ISO-8859-1', newline='\n' if sys.platform == 'win32' else None)
    else:
        open_file = open(fp)

    with open_file as ply:
        length = 0
        prop = []
        dtype_map = {'uint16': 'uint16', 'uint8': 'uint8', 'double': 'd', 'float32': 'f8', 
                     'float32': 'f4', 'float': 'f4', 'uchar': 'B', 'int': 'i'}
        dtype = []
        fmt = 'binary'
    
        for i, line in enumerate(ply.readlines()):
            length += len(line)
            if i == 1:
                if 'ascii' in line:
                    fmt = 'ascii'
            if 'element vertex' in line:
                N = int(line.split()[2])
            if 'property' in line:
                dtype.append(dtype_map[line.split()[1]])
                prop.append(line.split()[2])
            if 'element face' in line:
                raise Exception('.ply appears to be a mesh')
            if 'end_header' in line:
                break
    
        ply.seek(length)

        if fmt == 'binary':
            arr = np.fromfile(ply, dtype=','.join(dtype))
        else:
            arr = np.loadtxt(ply)
        
        df = pd.DataFrame(data=arr)
        df.columns = prop

        # Convert only x, y, z columns to float32
        for col in ['x', 'y', 'z']:
            if col in df.columns:
                df[col] = df[col].astype('float32')

    return df

def write_ply(output_name, pc, comments=[]):

    # -- 1) Drop duplicate columns from the DataFrame --
    pc = pc.loc[:, ~pc.columns.duplicated()].copy()

    # Force these columns to be float64 for consistency
    cols = ['x', 'y', 'z']
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']].astype('f8')

    with open(output_name, 'w') as ply:

        ply.write("ply\n")
        ply.write('format binary_little_endian 1.0\n')
        ply.write("comment Author: Phil Wilkes\n")
        for comment in comments:
            ply.write(f"comment {comment}\n")
        ply.write("obj_info generated with pcd2ply.py\n")
        ply.write(f"element vertex {len(pc)}\n")
        ply.write("property float64 x\n")
        ply.write("property float64 y\n")
        ply.write("property float64 z\n")

        # If 'red' is in pc columns, assume 'green','blue' are also present
        if 'red' in pc.columns:
            cols += ['red', 'green', 'blue']
            pc[['red', 'green', 'blue']] = pc[['red', 'green', 'blue']].astype('i')
            ply.write("property int red\n")
            ply.write("property int green\n")
            ply.write("property int blue\n")

        # -- 2) Dynamically detect additional columns --
        for col in pc.columns:
            # Skip if already in the 'cols' list:
            if col in cols:
                continue

            # Attempt casting to float64 (skip if it fails)
            try:
                pc[col] = pc[col].astype('f8')
                ply.write(f"property float64 {col}\n")
                cols.append(col)
            except:
                pass

        ply.write("end_header\n")

    with open(output_name, 'ab') as ply:
        # pc[cols] must have unique column names
        arr = pc[cols].to_records(index=False)
        ply.write(arr.tobytes())

        
# def write_ply(output_name, pc, comments=None):
#     """
#     Write point cloud data to PLY file with specific format and data types.
    
#     Args:
#         output_name (str): Output file path
#         pc (pd.DataFrame): Point cloud data
#         comments (list): Optional list of comments to include in header
#     """
#     if comments is None:
#         comments = []
    
#     # Create a copy to avoid modifying the original DataFrame
#     pc = pc.copy()
    
#     # Define expected columns and their data types
#     column_types = {
#         'x': ('double', 'float32'),
#         'y': ('double', 'float32'),
#         'z': ('double', 'float32'),
#         'time': ('double', 'float32'),
#         'nx': ('float', 'float32'),
#         'ny': ('float', 'float32'),
#         'nz': ('float', 'float32'),
#         'red': ('uchar', 'uint8'),
#         'green': ('uchar', 'uint8'),
#         'blue': ('uchar', 'uint8'),
#         'alpha': ('uchar', 'uint8'),
#         'label':  ('uchar', 'uint8'),
#     }
    
#     # If we have duplicate columns, keep the last occurrence (which should be the new alpha)
#     pc = pc.loc[:, ~pc.columns.duplicated(keep='last')]
    
#     # Convert columns to appropriate data types
#     for col, (_, dtype) in column_types.items():
#         if col in pc.columns:
#             pc[col] = pc[col].astype(dtype)
    
#     # Write header
#     with open(output_name, 'w') as ply:
#         ply.write("ply\n")
#         ply.write("format binary_little_endian 1.0\n")
#         ply.write("comment generated by raycloudtools library\n")
        
#         # Add any additional comments
#         for comment in comments:
#             ply.write(f"comment {comment}\n")
            
#         # Write number of vertices
#         ply.write(f"element vertex {len(pc)}\n")
        
#         # Write properties in specified order
#         cols_to_write = []
#         for col, (ply_type, _) in column_types.items():
#             if col in pc.columns:
#                 ply.write(f"property {ply_type} {col}\n")
#                 cols_to_write.append(col)
        
#         ply.write("end_header\n")
    
#     # Write binary data - only use columns we wrote to header
#     with open(output_name, 'ab') as ply:
#         ply.write(pc[cols_to_write].to_records(index=False).tobytes())
        
class dict2class:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def make_folder_structure(params):
    
    if params.odir == None:
        params.odir = os.path.join(params.directory, params.filename + '_FSCT_output')
    
    params.working_dir = os.path.join(params.odir, params.basename + '.tmp')
    
    if not os.path.isdir(params.odir):
        os.makedirs(params.odir)

    if not os.path.isdir(params.working_dir):
        os.makedirs(params.working_dir)
    else:
        shutil.rmtree(params.working_dir, ignore_errors=True)
        os.makedirs(params.working_dir)
        
    if params.verbose:
        print('output directory:', params.odir)
        print('scratch directory:', params.working_dir)
    
    return params
    
def voxelise(tmp, length, method='random', z=True):

    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    if z: tmp.loc[:, 'zz'] = tmp.z // length * length

    if method == 'random':
            
        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
            
        xD = {x:code() for x in tmp.xx.unique()}
        yD = {y:code() for y in tmp.yy.unique()}
        if z: zD = {z:code() for z in tmp.zz.unique()}
            
        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD) 
        if z: tmp.VX += tmp.zz.map(zD)
   
    elif method == 'bytes':
        
        code = lambda row: np.array([row.xx, row.yy] + [row.zz] if z else []).tobytes()
        tmp.loc[:, 'VX'] = tmp.apply(code, axis=1)
        
    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')
 
    return tmp 


def downsample(pc, vlength, 
               accurate=False,
               keep_columns=[], 
               keep_points=False,
               voxel_method='random', 
               return_VX=False,
               verbose=False):

    """
    Downsamples a point cloud so that there is one point per voxel.
    Points are selected as the point closest to the median xyz value
    
    Parameters
    ----------
    
    pc: pd.DataFrame with x, y, z columns
    vlength: float
    
    
    Returns
    -------
    
    pd.DataFrame with boolean downsample column
    
    """

    pc = pc.drop(columns=[c for c in ['downsample', 'VX'] if c in pc.columns])   
    columns = pc.columns.to_list() + keep_columns # required for tidy up later
    if return_VX: columns += ['VX']
    pc = voxelise(pc, vlength, method=voxel_method)

    if accurate:
        # groubpy to find central (closest to median) point
        groupby = pc.groupby('VX')
        pc.loc[:, 'mx'] = groupby.x.transform(np.median)
        pc.loc[:, 'my'] = groupby.y.transform(np.median)
        pc.loc[:, 'mz'] = groupby.z.transform(np.median)
        pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].to_numpy(dtype=np.float32) - 
                                           pc[['mx', 'my', 'mz']].to_numpy(dtype=np.float32), axis=1)
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True

    else:
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.VX.duplicated(), 'downsample'] = True
        
    if keep_points:
        return pc[columns + ['downsample']]
    else:
        return pc.loc[pc.downsample][columns]
    
def compute_plot_centre(pc):
    """calculate plot centre"""
    plot_min, plot_max = pc[['x', 'y']].min(), pc[['x', 'y']].max()
    return (plot_min + ((plot_max - plot_min) / 2)).values

def compute_bbox(pc):
    bbox_min = pc.min().to_dict()
    bbox_min = {k + 'min':v for k, v in bbox_min.items()}
    bbox_max = pc.max().to_dict()
    bbox_max = {k + 'max':v for k, v in bbox_max.items()}
    return dict2class({**bbox_min, **bbox_max})
    
def load_file(filename, additional_headers=False, verbose=False):
    
    file_extension = os.path.splitext(filename)[1]
    headers = ['x', 'y', 'z']

    if file_extension == '.las' or file_extension == '.laz':

        import laspy

        inFile = laspy.read(filename)
        pc = np.vstack((inFile.x, inFile.y, inFile.z))
        #for header in additional_fields:
        #    if header in list(inFile.point_format.dimension_names):
        #        pc = np.vstack((pc, getattr(inFile, header)))
        #    else:
        #        headers.drop(header)
        pc = pd.DataFrame(data=pc, columns=['x', 'y', 'z'])

    elif file_extension == '.ply':
        pc = read_ply(filename)
        #pc = pc[headers]
    else:
        raise Exception('point cloud format not recognised' + filename)

    original_num_points = len(pc)
    
    if verbose: print(f'read in {filename} with {len(pc)} points')
   
    if additional_headers:
        return pc, [c for c in pc.columns if c not in ['x', 'y', 'z']]
    else: return pc

    
def save_file(filename, pointcloud, additional_fields=[], verbose=False):
    """
    Save point cloud data to various file formats (.las, .csv, .ply)
    
    Args:
        filename (str): Output filename with extension
        pointcloud (numpy.ndarray or pandas.DataFrame): Point cloud data
        additional_fields (list): List of additional column names beyond x,y,z
        verbose (bool): Whether to print status messages
    """
    if pointcloud.shape[0] == 0:
        print(f"{filename} is empty...")
        return
        
    if verbose:
        print('Saving file:', filename)
        
    # Ensure additional_fields only contains columns that exist in the data
    if isinstance(pointcloud, pd.DataFrame):
        available_cols = pointcloud.columns.tolist()
        additional_fields = [col for col in additional_fields if col in available_cols]
    
    cols = ['x', 'y', 'z'] + additional_fields
    
    try:
        if filename.endswith('.las'):
            las = laspy.create(file_version="1.4", point_format=7)
            las.header.offsets = np.min(pointcloud[:, :3], axis=0)
            las.header.scales = [0.001, 0.001, 0.001]
            
            las.x = pointcloud[:, 0]
            las.y = pointcloud[:, 1]
            las.z = pointcloud[:, 2]
            
            if len(additional_fields) > 0:
                # Skip x,y,z columns if they're in additional_fields
                fields_to_add = [f for f in additional_fields if f not in ['x', 'y', 'z']]
                
                # The reverse step puts the headings in the preferred order
                col_idxs = list(range(3, pointcloud.shape[1]))
                fields_to_add.reverse()
                col_idxs.reverse()
                
                for header, i in zip(fields_to_add, col_idxs):
                    column = pointcloud[:, i]
                    if header in ['red', 'green', 'blue']:
                        setattr(las, header, column)
                    else:
                        las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                        setattr(las, header, column)
                        
            las.write(filename)
            if verbose:
                print("Saved.")
                
        elif filename.endswith('.csv'):
            if isinstance(pointcloud, pd.DataFrame):
                pointcloud.to_csv(filename, header=None, index=None, sep=' ')
            else:
                pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
            if verbose:
                print("Saved to:", filename)
                
        elif filename.endswith('.ply'):
            if not isinstance(pointcloud, pd.DataFrame):
                # Convert to DataFrame with proper column names
                cols = ['x', 'y', 'z'] + additional_fields
                pointcloud = pd.DataFrame(pointcloud, columns=cols[:pointcloud.shape[1]])
            
            # Ensure we only try to save columns that exist
            valid_cols = [col for col in cols if col in pointcloud.columns]
            write_ply(filename, pointcloud[valid_cols])
            if verbose:
                print("Saved to:", filename)
                
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
    except Exception as e:
        print(f"Error saving file {filename}: {str(e)}")
        raise

def make_dtm(params):
    
    """ 
    This function will generate a Digital Terrain Model (dtm) based on the terrain labelled points.
    """

    if params.verbose: print("Making dtm...")

    params.grid_resolution = .5

    ### voxelise, identify lowest points and create DTM
    params.pc = voxelise(params.pc, params.grid_resolution, z=False)
    VX_map = params.pc.loc[~params.pc.VX.duplicated()][['xx', 'yy', 'VX']]
    ground = params.pc.loc[params.pc.label == params.terrain_class] 
    ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.median)
    ground = ground.loc[ground.z == ground.zmin]
    ground = ground.loc[~ground.VX.duplicated()]

    X, Y = np.meshgrid(np.arange(params.pc.xx.min(), params.pc.xx.max() + params.grid_resolution, params.grid_resolution),
                       np.arange(params.pc.yy.min(), params.pc.yy.max() + params.grid_resolution, params.grid_resolution))

    ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy']) 
    ground_arr = pd.merge(ground_arr, VX_map, on=['xx', 'yy'], how='outer') # map VX to ground_arr
    ground_arr = pd.merge(ground[['z', 'VX']], ground_arr, how='right', on=['VX']) # map z to ground_arr
    ground_arr.sort_values(['xx', 'yy'], inplace=True)
    
    # loop over incresing size of window until no cell are nan
    ground_arr.loc[:, 'ZZ'] = np.nan
    size = 3 
    while np.any(np.isnan(ground_arr.ZZ)):
        ground_arr.loc[:, 'ZZ'] = ndimage.generic_filter(ground_arr.z.values.reshape(*X.shape), # create raster, 
                                                         lambda z: np.nanmedian(z), size=size).flatten()
        size += 2

    ground_arr[['xx', 'yy', 'ZZ']].to_csv(os.path.join(params.odir, f'{params.basename}.dem.csv'), index=False)

    # apply to all points   
    MAP = ground_arr.set_index('VX').ZZ.to_dict()
    params.pc.loc[:, 'n_z'] = params.pc.z - params.pc.VX.map(MAP)  
    
    return params


def chunk_pc(pc, out_dir, params):

    def save_pts(pc, I, bx, by, bz, working_dir, params):

        pc = pc.loc[(pc.x.between(bx, bx + 6)) &
                    (pc.y.between(by, by + 6)) &
                    (pc.z.between(bz, bz + 6))]

        if len(pc) > 1000:

            if len(pc) > 20000:
                pc = pc.sample(n=20000)

            np.save(os.path.join(working_dir, f'{I:07}'), pc[['x', 'y', 'z', 'label']].values)
    
    if not os.path.isdir(out_dir): os.makedirs(out_dir)
    
    # apply global shift
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']] - pc[['x', 'y', 'z']].mean()

    pc.reset_index(inplace=True)
    pc.loc[:, 'pid'] = pc.index

    # generate bounding boxes
    xmin, xmax = np.floor(pc.x.min()), np.ceil(pc.x.max())
    ymin, ymax = np.floor(pc.y.min()), np.ceil(pc.y.max())
    zmin, zmax = np.floor(pc.z.min()), np.ceil(pc.z.max())

    box_dims=6
    box_overlap=0.5
    
    box_overlap = box_dims * box_overlap

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)
    
    # multithread segmenting points into boxes and save
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts, args=(pc, i, bx, by, bz, out_dir, params)))

    for x in tqdm(threads, desc='generating data blocks', disable=False if params.verbose else True):
        x.start()

    for x in threads:
        x.join()

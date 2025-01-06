import os
import time
import shutil
import sys
import glob

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from abc import ABC
import torch
from torch_geometric.data import Dataset, DataLoader, Data
from fsct.model import Net

from tools import save_file

sys.setrecursionlimit(10 ** 8) # Can be necessary for dealing with large point clouds.

class TestingDataset(Dataset, ABC):
    def __init__(self, root_dir, points_per_box, device):
        self.filenames = glob.glob(os.path.join(root_dir, '*.npy'))
        self.device = device
        self.points_per_box = points_per_box

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        pos = point_cloud[:, :3]
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)

        # Place sample at origin
        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos = pos - local_shift
        data = Data(pos=pos, x=None, local_shift=local_shift)
        return data
    
def SemanticSegmentation(params):
    # if xyz is in global coords (e.g. when re-running) reset
    # coords to mean pos - required for accurate running of torch
    if not np.all(np.isclose(params.pc.loc[~params.pc.buffer][['x', 'y', 'z']].mean(), [0, 0, 0], atol=.1)):
        params.pc[['x', 'y', 'z']] -= params.global_shift

    if params.verbose:
        print('----- semantic segmentation started -----')
    
    params.sem_seg_start_time = time.time()
    
    # check status of GPU
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.verbose:
        print('using:', params.device)
    
    # generates pytorch dataset iterable
    test_dataset = TestingDataset(
        root_dir=params.working_dir,
        points_per_box=params.max_points_per_box,
        device=params.device
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=params.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # initialize model
    model = Net(num_classes=4).to(params.device)
    model.load_state_dict(torch.load(params.model, map_location=params.device), strict=False)
    model.eval()
    
    with torch.no_grad():
        output_point_cloud = np.zeros((0, 3 + 4))
        output_list = []
        
        for data in tqdm(test_loader, disable=False if params.verbose else True):
            data = data.to(params.device)
            out = model(data)
            out = out.permute(2, 1, 0).squeeze()
            batches = np.unique(data.batch.cpu())
            out = torch.softmax(out.cpu().detach(), axis=1)
            pos = data.pos.cpu()
            output = np.hstack((pos, out))
            
            for batch in batches:
                outputb = np.asarray(output[data.batch.cpu() == batch])
                outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch:3 + (3 * batch)]
                output_list.append(outputb)
        
        classified_pc = np.vstack(output_list)
        
        # clean up anything no longer needed to free RAM
        del outputb, out, batches, pos, output
        
        if params.verbose:
            print("Matching points to original cloud...")
            
        # Find nearest neighbors for each point in the original point cloud
        neighbours = NearestNeighbors(
            n_neighbors=1,  # Only need closest point
            algorithm='kd_tree',
            metric='euclidean'
        ).fit(classified_pc[:, :3])
        
        # Get indices of nearest points
        _, indices = neighbours.kneighbors(params.pc[['x', 'y', 'z']].values)
        
        # Get wood probabilities for matched points (last column, -1)
        wood_probs = classified_pc[indices[:, 0], -1]
        
        # Convert to 0-255 range for uint8
        alpha_values = (wood_probs * 255).astype(np.uint8)
        
        # Find smallest non-zero value in alpha
        min_nonzero = alpha_values[alpha_values > 0].min() if np.any(alpha_values > 0) else 1
        
        # Replace zeros with smallest non-zero value
        alpha_values[alpha_values == 0] = min_nonzero
        
        # Assign alpha values to the point cloud
        params.pc['alpha'] = alpha_values
        
        # shift back to global coords
        params.pc[['x', 'y', 'z']] += params.global_shift
        
        save_file(
            os.path.join(params.odir, '{}.segmented.{}'.format(params.filename[:-4], params.output_fmt)),
            params.pc.loc[~params.pc.buffer],
            additional_fields=['alpha'] + params.additional_headers
        )
        
        params.sem_seg_total_time = time.time() - params.sem_seg_start_time
        if not params.keep_npy:
            [os.unlink(f) for f in test_dataset.filenames]
            
        print("semantic segmentation done in", params.sem_seg_total_time, 's\n')
        
    return params
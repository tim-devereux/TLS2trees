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
                # Shift positions back
                shift_vals = np.asarray(data.local_shift.cpu())[3*batch : 3 + (3*batch)]
                outputb[:, :3] = outputb[:, :3] + shift_vals
                output_list.append(outputb)

        classified_pc = np.vstack(output_list)

    # clean up anything no longer needed to free RAM
    del outputb, out, batches, pos, output

    # choose most confident label
    if params.verbose:
        print("Choosing most confident labels...")

    # Fit neighbors once on the classified points.
    neighbours = NearestNeighbors(
        n_neighbors=16, 
        algorithm='kd_tree', 
        metric='euclidean', 
        radius=0.05
    ).fit(classified_pc[:, :3])

    # Remove old label columns if they exist
    params.pc = params.pc.drop(columns=[c for c in params.pc.columns if c in ['label', 'pWood']], errors='ignore')

    # Prepare an array to store label probabilities (or medians)
    num_points = params.pc.shape[0]
    labels = np.zeros((num_points, 4), dtype=np.float32)

    # Define a chunk size that fits comfortably into RAM.
    chunk_size = 100000

    # Query neighbors in small batches:
    for start_idx in range(0, num_points, chunk_size):
        end_idx = min(start_idx + chunk_size, num_points)
        
        chunk_xyz = params.pc.iloc[start_idx:end_idx][['x', 'y', 'z']].values
        
        _, chunk_indices = neighbours.kneighbors(chunk_xyz)
        
        # classified_pc has shape [N_classified, >=3+4]
        # columns = X, Y, Z, and last 4 columns are class probabilities
        chunk_probs = classified_pc[chunk_indices][:, :, -4:]
        
        # Compute the median probability across k neighbors
        chunk_medians = np.median(chunk_probs, axis=1)
        
        # Store medians in our labels array
        labels[start_idx:end_idx] = chunk_medians

    print("labels chosen")

    # Choose label = argmax across the 4 columns
    params.pc['label'] = np.argmax(labels, axis=1)

    # pWood is the last column (index 3) in the 4 columns
    params.pc['pWood'] = labels[:, -1]

    print("applying global shift")

    # Shift back to global coords if needed
    params.pc[['x', 'y', 'z']] += params.global_shift

    # Now set alpha = pWood
    params.pc['alpha'] = params.pc['pWood']

    # ---------------------------------------------------------------------
    # Save two files:
    #   1) One with alpha == pWood
    #   2) One with labels and pWood
    # ---------------------------------------------------------------------
    
    # 1) File with alpha == pWood
    df_alpha = params.pc.loc[~params.pc.buffer].copy()
    # Remove duplicate columns if they exist
    df_alpha = df_alpha.loc[:, ~df_alpha.columns.duplicated()]
    
    save_file(
        os.path.join(params.odir, '{}.segmented_alpha.{}'.format(params.filename[:-4], params.output_fmt)),
        df_alpha,
        additional_fields=['alpha'] + params.additional_headers
    )

    # 2) File with labels and pWood
    df_labels = params.pc.loc[~params.pc.buffer].copy()
    # Remove duplicate columns if they exist
    df_labels = df_labels.loc[:, ~df_labels.columns.duplicated()]
    
    save_file(
        os.path.join(params.odir, '{}.segmented_labels.{}'.format(params.filename[:-4], params.output_fmt)),
        df_labels,
        additional_fields=['label',] + params.additional_headers
    )
    # ---------------------------------------------------------------------

    params.sem_seg_total_time = time.time() - params.sem_seg_start_time
    if not params.keep_npy:
        [os.unlink(f) for f in test_dataset.filenames]

    print("semantic segmentation done in", params.sem_seg_total_time, 's\n')

    return params

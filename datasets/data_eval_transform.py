from torch.utils.data import Dataset
import numpy as np
import os
import math
from PIL import Image
from datasets.data_io import *

from datasets.preprocess import *

# Test any dataset with scale and center crop

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=True,
                adaptive_scaling=True, max_h=1200,max_w=1600,sample_scale=1,base_image_size=8,light_idx=3,**kwargs):
        super(MVSDataset, self).__init__()
        
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth
        self.light_idx = light_idx

        self.adaptive_scaling=adaptive_scaling
        self.max_h=max_h
        self.max_w=max_w
        self.sample_scale=sample_scale
        self.base_image_size=base_image_size
        
        assert self.mode in ["test", "val"]
        self.metas = self.build_list()
        print('Data Loader : data_eval_transform **************' )

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()   
            scans = [line.rstrip() for line in scans]

        # scans - use training data structure: Cameras/pair.txt (shared, not per scan)
        pair_file = "Cameras/pair.txt"
        # read the pair file
        with open(os.path.join(self.datapath, pair_file)) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                # For each scan, add the same view pairs
                for scan in scans:
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval


    def read_img(self, filename):
        img = Image.open(filename)
        # Convert to RGB if image has alpha channel (RGBA -> RGB)
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        mat=np.array(img, dtype=np.float32)
        
        return self.center_img(mat)	

    def center_img(self, img): # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0,1), keepdims=True)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        return (img - mean) / (np.sqrt(var) )

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        
        if self.nviews>len(src_views):
              self.nviews=len(src_views)+1
              
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        cams=[]
        extrinsics_list=[]
        

        # Load depth first to get target dimensions
        ref_view = view_ids[0]
        depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, ref_view))
        depth = self.read_depth(depth_filename)
        depth_h, depth_w = depth.shape[:2]
        
        # Get camera params for reference view to compute depth values
        ref_proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt'.format(ref_view))
        ref_intrinsics, ref_extrinsics, depth_min, depth_interval = self.read_cam_file(ref_proj_mat_filename)
        
        for i, vid in enumerate(view_ids):
            # Use training data structure: Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png
            # Note: vid is 0-indexed (0-48) but filenames use 1-indexed (1-49), so use vid+1
            img_filename = os.path.join(self.datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, self.light_idx))
            # Use training data structure: Cameras/train/{vid:08d}_cam.txt
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt'.format(vid))

            img = self.read_img(img_filename)
            # Resize image to match depth map dimensions
            import cv2
            img_h, img_w = img.shape[:2]
            if img_h != depth_h or img_w != depth_w:
                img_resized = cv2.resize(img, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
                # Scale camera intrinsics to match image resize
                scale_h = depth_h / img_h
                scale_w = depth_w / img_w
                intrinsics, extrinsics, _, _ = self.read_cam_file(proj_mat_filename)
                intrinsics[0, 0] *= scale_w  # fx
                intrinsics[1, 1] *= scale_h  # fy
                intrinsics[0, 2] *= scale_w  # cx
                intrinsics[1, 2] *= scale_h  # cy
            else:
                img_resized = img
                intrinsics, extrinsics, _, _ = self.read_cam_file(proj_mat_filename)
            
            imgs.append(img_resized)
            cams.append(intrinsics)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics_list.append(extrinsics)
            
            if i == 0:  # reference view
                
                if self.inverse_depth: #slice inverse depth
                    print('Process {} inverse depth'.format(idx))
                    depth_end = depth_interval * (self.ndepths-1) + depth_min # wether depth_end is this
                    depth_values = np.linspace(1.0 / depth_min, 0.0, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                else:
                    depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval ,
                                            dtype=np.float32) 
                                                                           
                    depth_end = depth_interval * self.ndepths + depth_min

        # Create mask from depth values
        mask = np.array((depth >= depth_min) & (depth <= depth_end), dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2]) # B,C,H,W
        
        # Transpose for processing: (views, H, W, C)
        imgs_processed = imgs.transpose(0, 2, 3, 1)
        
        # Ensure dimensions are divisible by base_image_size (required by network)
        current_h, current_w = imgs_processed[0].shape[0], imgs_processed[0].shape[1]
        target_h = int(math.ceil(current_h / self.base_image_size) * self.base_image_size)
        target_w = int(math.ceil(current_w / self.base_image_size) * self.base_image_size)
        
        # Resize images, depth, and mask to target dimensions if needed
        if current_h != target_h or current_w != target_w:
            import cv2
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            resized_imgs = []
            for view in range(self.nviews):
                img = imgs_processed[view]
                resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_imgs.append(resized)
                # Update camera intrinsics
                cams[view][0, 0] *= scale_w  # fx
                cams[view][1, 1] *= scale_h  # fy
                cams[view][0, 2] *= scale_w  # cx
                cams[view][1, 2] *= scale_h  # cy
            imgs_processed = np.array(resized_imgs)
            # Resize depth and mask
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Transpose back to (views, C, H, W)
        croped_imgs = imgs_processed.transpose(0, 3, 1, 2)
        croped_depth = depth
        croped_mask = mask

        new_proj_matrices = []
        for id in range(self.nviews):
            proj_mat = extrinsics_list[id]
            proj_mat[:3, :4] = np.matmul(cams[id], proj_mat[:3, :4])
            new_proj_matrices.append(proj_mat)

        new_proj_matrices = np.stack(new_proj_matrices)

        result = {"imgs": croped_imgs,
                "proj_matrices": new_proj_matrices,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
        
        # Add depth and mask if available
        if croped_depth is not None:
            result["depth"] = croped_depth
            result["mask"] = croped_mask
        
        return result

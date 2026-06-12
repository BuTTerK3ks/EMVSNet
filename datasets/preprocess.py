#!/usr/bin/env python
import math
import cv2
import numpy as np


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    
    # focal: 
    new_cam[0][0] = cam[0][0] * scale
    new_cam[1][1] = cam[1][1] * scale
    # principle point:
    new_cam[0][2] = cam[0][2] * scale
    new_cam[1][2] = cam[1][2] * scale
    return new_cam

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def scale_mvs_input(images, cams, depth_image=None, scale=1,view_num=5):
    """ resize input to fit into the memory """
    new_images = []
    new_cams=[]
    for view in range(view_num):
        new_images.append(scale_image(images[view], scale=scale))
        new_cams.append(scale_camera(cams[view], scale=scale))
    new_images = np.array(new_images)
    if depth_image is None:
        #return images, cams
        return new_images, new_cams
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        return new_images, cams, depth_image

def crop_mvs_input(images, cams, depth_image=None,view_num=5,max_h=1200,max_w=1600,base_image_size=8):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    
    # First, determine the target crop size based on the first image (all views should have same size after scaling)
    h, w = images[0].shape[0:2]
    new_h = h
    new_w = w
    if new_h > max_h:
        new_h = max_h
    else:
        new_h = int(math.ceil(h /base_image_size) * base_image_size)
    if new_w > max_w:
        new_w = max_w
    else:
        new_w = int(math.ceil(w /base_image_size) * base_image_size)
    
    # Compute crop parameters once (use first image dimensions as reference)
    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = int(start_h + new_h)
    finish_w = int(start_w + new_w)
    
    new_images = []
    # crop images and cameras - use same crop parameters for all views to ensure consistent sizes
    for view in range(view_num):
        view_h, view_w = images[view].shape[0:2]
        # Use the same crop parameters, but ensure we don't exceed bounds
        actual_start_h = max(0, min(start_h, view_h - new_h))
        actual_start_w = max(0, min(start_w, view_w - new_w))
        actual_finish_h = min(finish_h, view_h)
        actual_finish_w = min(finish_w, view_w)
        
        # Crop the image
        cropped = images[view][actual_start_h:actual_finish_h, actual_start_w:actual_finish_w]
        
        # Ensure all views have exactly the same size (new_h x new_w)
        # If cropped size doesn't match, pad or crop further
        if cropped.shape[0] < new_h or cropped.shape[1] < new_w:
            # Pad if smaller
            pad_h = max(0, new_h - cropped.shape[0])
            pad_w = max(0, new_w - cropped.shape[1])
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            if len(cropped.shape) == 3:
                cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
            else:
                cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        elif cropped.shape[0] > new_h or cropped.shape[1] > new_w:
            # Crop if larger (shouldn't happen, but just in case)
            cropped = cropped[:new_h, :new_w]
        
        new_images.append(cropped)
        cams[view][0][2] = cams[view][0][2] - actual_start_w
        cams[view][1][2] = cams[view][1][2] - actual_start_h

    new_images = np.stack(new_images)
    # crop depth image
    if not depth_image is None:
        # Ensure indices are Python ints for slicing
        depth_image = depth_image[int(start_h):int(finish_h), int(start_w):int(finish_w)]
        return new_images, cams, depth_image
    else:
        return new_images, cams


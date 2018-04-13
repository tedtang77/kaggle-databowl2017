import os, sys, math
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import scipy

from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.segmentation import clear_border

import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_filename(case, file_list):
    '''
        Helper function to get rows in data frame associated with each file
    '''
    for f in file_list:
        if case in f:
            return(f)
        

def get_images_and_masks(fcount, img_file, mini_df, path, resampling=True): 
    # load the data once
    itk_img = sitk.ReadImage(img_file) 
    img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    num_z, height, width = img_array.shape      # heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # x,y,z  Spacing of voxels in world coor. (mm)
    
    # For Resampling
    if resampling is True:
        img_array, spacing = resample(img_array, spacing[::-1]) # [::-1] turn (x,y,z) spacing into (z,y,x) 
        spacing = spacing[::-1] # [::-1] turn (z,y,x) spacing back into (x,y,z) 
        num_z, height, width = img_array.shape
        
    for node_idx, cur_row in mini_df.iterrows():       
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
            
        # just keep 3 slices
        imgs = np.ndarray([3,height,width],dtype=np.int16) #dtype=np.float32) # np.int16 as imgs dtype
        masks = np.ndarray([3,height,width],dtype=np.int8) #dtype=np.uint8) # np.int8 as imgs dtype
        center = np.array([node_x,node_y,node_z])   # nodule center
        v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)

        for i, i_z in enumerate(range(int(v_center[2])-1,int(v_center[2])+2)):
            mask = make_mask(center, diam, i_z*spacing[2]+origin[2], width, height, spacing, origin)
            masks[i] = mask
            #imgs[i] = matrix2uint16(img_array[i_z])
            imgs[i] = img_array[i_z]

        
        np.save(os.path.join(path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)
        np.save(os.path.join(path,"spacing_%04d_%04d.npy" % (fcount, node_idx)),spacing)
    
'''    
def get_images_and_masks_resample(fcount, img_file, mini_df, path): 
    # load the data once
    itk_img = sitk.ReadImage(img_file) 
    img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    num_z, height, width = img_array.shape      # heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # x,y,z  Spacing of voxels in world coor. (mm)
    
    # For Resampling  
    img_array_resampled, new_spacing = resample(img_array, spacing[::-1]) # [::-1] turn (x,y,z) spacing into (z,y,x) 
    new_spacing = new_spacing[::-1] # [::-1] turn (z,y,x) spacing back into (x,y,z) 
    num_z_resampled, height_resampled, width_resampled = img_array_resampled.shape
        
    for node_idx, cur_row in mini_df.iterrows():       
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
            
        # just keep 3 slices
        imgs_resampled = np.ndarray([3,height_resampled,width_resampled],dtype=np.int16) #dtype=np.float32) # np.int16 as imgs dtype
        masks_resampled = np.ndarray([3,height_resampled,width_resampled],dtype=np.int8) #dtype=np.uint8) # np.int8 as imgs dtype
        center = np.array([node_x,node_y,node_z])   # nodule center
        v_center_resampled =np.rint((center-origin)/new_spacing)  # nodule center in voxel space (still x,y,z ordering)
            
        for i, i_z in enumerate(range(int(v_center_resampled[2])-1,int(v_center_resampled[2])+2)):
            mask_resampled = make_mask(center, diam, i_z*new_spacing[2]+origin[2], width_resampled, height_resampled, new_spacing, origin)
            masks_resampled[i] = mask_resampled
            #imgs[i] = matrix2uint16(img_array[i_z])
            imgs_resampled[i] = img_array_resampled[i_z]
                
        np.save(os.path.join(path,"images_resampled_%04d_%04d.npy" % (fcount, node_idx)), imgs_resampled)
        np.save(os.path.join(path,"masks_resampled_%04d_%04d.npy" % (fcount, node_idx)), masks_resampled)
        np.save(os.path.join(path,"newspacing_%04d_%04d.npy" % (fcount, node_idx)), new_spacing)
'''

def make_mask(center, diam, z, width, height, spacing, origin):
    '''
        the mask coordinates have to match the ordering of the array coordinates. 
        The x and y ordering is flipped (in (y, x) ordering)
        Args:
            Center : centers of circles px -- list of coordinates x,y,z
            diam : diameters of circles px -- diameter
            widthXheight : pixel dim of image
            spacing = mm/px conversion rate np array x,y,z
            origin = x,y,z mm np.array
            z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)


def resample(image, spacing, new_spacing=[1,1,1]):
    '''
        resampling the full dataset to a certain isotropic resolution, 1mm*1mm*1mm pixels. So we can use 3D convnets without worrying about learning zoom/slice thickness invariance.
        
        A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], this can be problematic for automatic analysis (e.g. using ConvNets)! 
        
        Note: When you apply this, to save the new spacing! Due to rounding this may be slightly off from the desired spacing (above script picks the best possible spacing with rounding).
        
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        
        Args:
            image- indexes are z,y,x (notice the ordering)
            spacing- indexes are z,y,x (notice the ordering)
            new_spacing- indexes are z,y,x (notice the ordering)
    '''
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='constant')#, mode='nearest')
    
    return image, new_spacing


def matrix2uint16(matrix):
    ''' 
        Args:
            matrix must be a numpy array NXN
        Returns: 
            uint16 version matrix
    '''
    
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))


def get_images_and_masks_by_nodule_position(subset_path, output_path, annotation_fname, resampling=True):
    '''
        Args:
            subset_path: the path which saves lung data subset path
            output_path: the path to output nodule images and mask files
            annotation_fname: the path to save nodule_position annotation.csv            
           
    '''
    file_list=glob(subset_path+"*.mhd")
    print('total .mhd file:', len(file_list))
    
    # The locations of the nodes
    df_node = pd.read_csv(annotation_fname)
    df_node["file"] = df_node["seriesuid"].apply(get_filename, args=(file_list,))
    df_node = df_node.dropna()
    
    for fcount, img_file in enumerate(file_list):
        print("Getting mask for image file %s" % img_file.rsplit('/', 1)[-1].rsplit('\\', 1)[-1])
        
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if len(mini_df)>0:       # some files may not have a nodule--skipping those 
            get_images_and_masks(fcount, img_file, mini_df, output_path, resampling)
        

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

    
def find_threshold(img):
    #Standardize the pixel values
    #mean = np.mean(img)
    #std = np.std(img)
    #img = img-mean
    #img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    height, width = img.shape[1], img.shape[2]
    middle = img[:,int(height*0.2):int(height*0.8),int(width*0.2):int(width*0.8)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (radio-opaque tissue)
    # and background (radio transparent tissue ie lungs)
    # Doing this only on the center of the image to avoid 
    # the non-tissue parts of the image as much as possible
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    
    return threshold    


def segment_lung_mask(image, fill_lung_structures=True):
    '''
        Lung segmentation
        It involves quite a few smart steps. It consists of a series of applications of region growing and morphological operations. In this case, we will use only connected component analysis.
        
        I then use some erosion and dilation to fill in the incursions into the lungs region by radio-opaque tissue, followed by a selection of the regions based on the bounding box sizes of each region. The initial set of regions looks like 

        The steps:  
            * Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
            * Do connected components, determine label of air around person, fill this with 1s in the binary image
            * Optionally: For every axial slice in the scan, determine the largest solid connected component (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
            * Keep only the largest air pocket (the human body has other pockets of air here and there).
            
        Note: **This segmentation may fail for some edge cases**. It relies on the fact that the air outside the patient is not connected to the air in the lungs. If the patient has a [tracheostomy](https://en.wikipedia.org/wiki/Tracheotomy), this will not be the case, I do not know whether this is present in the dataset. Also, particulary noisy images (for instance due to a pacemaker in the image below) this method may also fail. Instead, the second largest air pocket in the body will be segmented. You can recognize this by checking the fraction of image that the mask corresponds to, which will be very small for this case. You can then first apply a morphological closing operation with a kernel a few mm in size to close these holes, after which it should work (or more simply, do not use the mask for this image). 
            
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    '''
    
    # Convert into a binary image by threshold
    threshold = find_threshold(image) # -400 # -320 
    threshold_image = np.array(image > threshold, dtype=np.int8)
    
    # Remove the blobs connected to the border of the image.
    #binary_image = clear_border(threshold_image)
    
    #binary_image = measure.label(binary_image)
    #
    # I found an initial erosion helful for removing graininess from some of the regions
    # and then large dialation is used to make the lung region 
    # engulf the vessels and incursions into the lung cavity by 
    # radio opaque tissue
    #
    #eroded_image = morphology.binary_erosion(threshold_image, np.ones([3,4,4])) # erosion(threshold_image,np.ones([4,4]))
    #dilation_image = morphology.binary_dilation(eroded_image, np.ones([3,10,10])) # dilation(eroded_image,np.ones([10,10]))
    dilation_image = morphology.binary_opening(threshold_image)
    #dilation_image = threshold_image
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = dilation_image+1 # np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    # Remove the blobs connected to the border of the image.
    #binary_image = clear_border(labels)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    #background_label = labels[0,0,0] 
    #Fill the air around the person
    #binary_image[background_label == labels] = 2
    for i in range(3):
        background_label = labels[i,0,0]
        binary_image[background_label == labels] = 2
        background_label = labels[i,-1,0]
        binary_image[background_label == labels] = 2
        background_label = labels[i,0,-1]
        binary_image[background_label == labels] = 2
        background_label = labels[i,-1,-1]
        binary_image[background_label == labels] = 2
    
    '''
    # Keep the labels with 2 largest areas.
    areas = [r.area for r in measure.regionprops(labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       binary_image[coordinates[0], coordinates[1]] = 2 #0 # label small area as background: 2
    '''
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    eroded_binary_image = morphology.binary_erosion(binary_image, np.ones([3,4,4])) 
    dilation_binary_image = morphology.binary_dilation(eroded_binary_image, np.ones([3,10,10])) 
    #dilation_binary_image = morphology.binary_opening(binary_image)
    binary_image = dilation_binary_image
        
    '''
    #TODO: This part of code causes problem 
    #      on LUNA dataset with removing some lung mask. 
    #      Need to study how to utilize it
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    print(l_max)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    '''

    return binary_image # dilation_binary_image 


def get_segment_lung_mask(path, fill_lung_structures=True):
    '''
        Isolation of the Lung Region of Interest to Narrow Our Nodule Search

        To isolate the lungs in the images. The general strategy is to threshold the image to isolate the 
        regions within the image, and then to identify which of those regions are the lungs. The lungs have a high constrast with the surrounding tissue, so the thresholding is fairly straightforward. We use some ad-hoc criteria for eliminating the non-lung regions from the image which do not apply equally to all data sets. 
    '''
    
    file_list=glob(path+"images*.npy")
    print('total images:', len(file_list))
    
    for img_file in file_list:
        imgs = np.load(img_file)
        print("on image", img_file)
        
        #height, width = imgs.shape[1], imgs.shape[2]
        #segmented_lungs_fill = np.ndarray([3,height,width],dtype=np.int8)
        #for i in range(3):
        #    segmented_lungs_fill[i] = segment_lung_mask(imgs[i], fill_lung_structures)
        segmented_lungs_fill = segment_lung_mask(imgs, fill_lung_structures)
        
        np.save(img_file.replace("images","lungmask"), segmented_lungs_fill)



def plot_3d(image, threshold=-300):
    '''
        3D plotting the scan.
        For visualization it is useful to be able to show a 3D image of the scan. Use marching cubes to create an approximate mesh for our 3D object, and plot this with matplotlib. Quite slow and ugly, but the best we can do.
        
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        
    '''
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    #verts, faces = measure.marching_cubes(p, threshold)
    verts, faces, _, _ = measure.marching_cubes(p, level=threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def plot_pixels(image, slice_idx=80):
    '''
       plot patient image pixels (before resampling)
       
       Args:
           pixels: patient image pixels
           slice_idx: the slice to show (middle slice is preferred)
       
    '''
    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    # Show some slice in the middle
    plt.imshow(image[slice_idx], cmap=plt.cm.gray)
    plt.show()


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def crop_lung_images_and_mask(path, new_size=[512, 512]):
    '''
        Args:
            path: the path which saves "lungmask*" files
            new_size: array. 
        Returns:
            out_images: list. final set of images 
            out_nodemasks: list. final set of nodemasks 
    '''
    file_list=glob(path+"lungmask*.npy");
    print('total lungmask:', len(file_list))
    
    out_images = []      #final set of images
    out_nodemasks = []   #final set of nodemasks
    for fname in file_list:
        print("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask","images"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask","masks"))
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            img_height, img_width = img.shape
            img= mask*img          # apply lung mask

            #make image bounding box  (min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            #
            # Finding the global min and max row over all regions
            #
            min_row = img_height # 512
            max_row = 0
            min_col = img_width # 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]
            width = max_col-min_col
            height = max_row - min_row
            if width > height:
                max_row=min_row+width
            else:
                max_col = min_col+height
            # 
            # cropping the image down to the bounding box for all regions
            # (there's probably an skimage command that can do this in one line)
            # 
            img = img[min_row:max_row,min_col:max_col]
            mask =  mask[min_row:max_row,min_col:max_col]
            if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
                pass
            else:
                new_img = resize(img, new_size, preserve_range=True, mode='constant')
                new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col], new_size, preserve_range=True, mode='constant')
                out_images.append(new_img)
                out_nodemasks.append(new_node_mask)
                
    return out_images, out_nodemasks          
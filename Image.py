
import numpy as np
from scipy import stats
import pickle
from PIL import Image as Pil_image, ImageTk
import cv2
from sklearn.cluster import MeanShift
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class Image:
    def __init__(self, img, res_shape):
        self.img = img
        self.min_contour_len = 20
        self.groups_cent = []
        self.resize(res_shape[0], res_shape[1])


    def resize(self, width, height):
        self.img_d = cv2.resize(self.img,
                                (width, height), 
                                interpolation = cv2.INTER_AREA)
        self.img_pil = ImageTk.PhotoImage(Pil_image.fromarray(self.img_d))


    def segment(self, points_per_side, pred_iou_thresh, min_contour_len):
        """
        Use Anysegment pretrained model to segment objects present in the image

        Arguments
        ----------
        points_per_side : int
            Number of points per each side of image, used for segmentation. 
            16-32, larger number segment more objects.
        pred_iou_thresh : float
            IoU thereshould for non-max suppresion.
        min_contour_len : int
            Threshould for length of the smallest contour kept.

        Returns
        -------
        list:
            Contours of segmented objects, used for plotting.

        """

        sam_checkpoint = 'sam_vit_b_01ec64.pth'
        model_type = 'vit_b'
        device = 'cuda'
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                    points_per_side=points_per_side, 
                                                    pred_iou_thresh=pred_iou_thresh
                                                    )    
        self.masks = mask_generator.generate(self.img_d)
        with open('masks_raw.pkl', 'wb') as file:
            pickle.dump(self.masks, file)
        # with open('masks_raw.pkl', 'rb') as file:
        #     self.masks = pickle.load(file)
        
        self.contours, mask_union = contours_fromMask(self.masks)
        bg = self.img_d[np.bool_(1-mask_union)]
        self.bg_val = stats.mode(bg, axis=0)[0].squeeze()
        self.min_contour_len = min_contour_len
        self.contours = list(filter(lambda x: len(x)>self.min_contour_len, self.contours))
            
        return self.contours


    def remove_object(self, point):
        """
        Remove object from image masks based on given coordinates. 
        
        Arguments
        ----------
        point : tuple
            Coodinates inside the object that should be deleted .

        Returns
        -------
        np.ndarray
            Contour of the removed object, used only for ploting.

        """
        contours_remove = []
        for i, mask in enumerate(self.masks):
            if mask['segmentation'][point.y, point.x] == True:            
                con, _ = cv2.findContours(mask['segmentation'].astype('uint8'),
                                                    cv2.RETR_LIST, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
                idx_large = np.argmax(np.array([len(x) for x in con]))
                contours_remove.append(con[idx_large])
                self.masks.pop(i)
        idx_large = np.argmax(np.array([len(x) for x in contours_remove]))    # becuase there are overlap between masks, only largers one will be passed as output
        return contours_remove[idx_large]


    def combine_objects(self, sqr):
        """
        Combine objects from image masks based on the given selection square.
        Groups_cent contains centre of group object and is updated every time the method is called

        Arguments
        ----------
        sqr : dict
            dictionary containing coordinates of a sqaure used for combinging the mask.

        Returns
        -------
        contours_group : list
            list of contours of the combined objects, used only for ploting.

        """    
        self.groups_cent.append([])
        contours_group = []
        for i, mask in enumerate(self.masks):
            if mask['segmentation'][sqr['y0']:sqr['y1'], sqr['x0']:sqr['x1']].sum() > 13:                    
                self.groups_cent[-1].append( np.mean(np.argwhere(mask['segmentation']), 0).astype(int) )                  
                con, _ = cv2.findContours(mask['segmentation'].astype('uint8'),
                                          cv2.RETR_LIST, 
                                          cv2.CHAIN_APPROX_SIMPLE)
                idx_large = np.argmax(np.array([len(x) for x in con]))
                contours_group.append(con[idx_large])
                
        return contours_group

    
    def sort_objects(self):
        """
        Finalize the combining request. Find contours centroid coordinates. Sort them left-right top-bottom.

        Returns
        -------
        contours: np.ndarray
        contour centroid coordinates: np.ndarray

        """         
        self.contours, _ = contours_fromMask(self.masks)
        self.contours = list(filter(lambda x: len(x)>self.min_contour_len, self.contours))        
        # combine requested contours
        if len(self.groups_cent)>0:
            for group in self.groups_cent:
                new_contour = []
                for i, con in enumerate(self.contours):
                    if any([cv2.pointPolygonTest(con, (int(cent[1]),int(cent[0])), False) >= 0 for cent in group]):
                        new_contour.append(con)
                        self.contours.pop(i)
                if len(new_contour)>0:
                    self.contours.append(new_contour)
        for i, con in enumerate(self.contours): # put eveything in a nested list
            if not isinstance(con, list):
                self.contours[i] = [con]
                  
        self.contours = np.array(self.contours, dtype=object)                     
        self.slice_no = len(self.contours)  
        self.contours_centroid = np.zeros((self.slice_no,2), dtype=int)
        for i, con_group in enumerate(self.contours):
            for con in con_group:
                M = cv2.moments(con)
                self.contours_centroid[i,0] += int(M['m01']/M['m00']) # row
                self.contours_centroid[i,1] += int(M['m10']/M['m00']) # col
            self.contours_centroid[i,0] = int(self.contours_centroid[i,0]/len(con_group))
            self.contours_centroid[i,1] = int(self.contours_centroid[i,1]/len(con_group))         
        
        # sort contours left-right top-bottom
        ms = MeanShift(bandwidth = len(self.img_d)/10, bin_seeding=True) # 5-30 would work for spliting centroid's x coordinates
        ms.fit(self.contours_centroid[:,1][:,np.newaxis])
        self.col_no = len(ms.cluster_centers_)
        self.row_no = self.slice_no/self.col_no
        contours_centroid_tmp = self.contours_centroid.copy()
        for i in range(self.col_no):  # replace x coor with cluster_centers
            contours_centroid_tmp[:,1][ms.labels_ == i] = ms.cluster_centers_[i]
        idx_cent_sort = np.lexsort(contours_centroid_tmp.T)   # first sort the colums, then rows
        self.contours = self.contours[idx_cent_sort]
        self.contours_centroid = self.contours_centroid[idx_cent_sort]
        
        return self.contours, self.contours_centroid
        


    def convert_to_stack(self, idx_slices, flips):
        """
        Convert image to stack of brain sections based on contours.

        Parameters
        ----------
        idx_slices : list
            Final order of brain slices after user visual inspection.

        Returns
        -------
        np.ndarray
            Image stack.

        """
        self.contours = self.contours[idx_slices]
        self.contours_centroid = self.contours_centroid[idx_slices] 
        # find coordinates of points inside each section
        masks_arg= []
        kernel = np.ones((5, 5), dtype='uint8')
        for con_group in self.contours:
            img_tmp = np.zeros_like(self.img_d[:,:,0])
            for j in range(len(con_group)):
                cv2.drawContours(img_tmp, con_group, j, color=255, thickness=-1)
            img_tmp = cv2.dilate(img_tmp, kernel, iterations=1)
            masks_arg.append(np.argwhere(img_tmp))
                
        height = int(1.2*self.img_d.shape[0]/self.row_no)   # 1.2 to add margin
        width = int(1.2*self.img_d.shape[1]/self.col_no)
        
        masks_arg_stack = []
        for i, mask in enumerate(masks_arg):    # transform section coordinates to new stack
            mask_arg_tmp = mask - self.contours_centroid[i] + np.array([height, width])//2
            masks_arg_stack.append(tuple(mask_arg_tmp.T))
        
        # tranfer each section to a new stack
        self.img_stack = np.empty((self.slice_no, height, width, 3), dtype='uint8')
        for i in range(3):      # replace rgb values for background
            self.img_stack[:,:,:,i] = self.bg_val[i]
        for i in range(self.slice_no):
            self.img_stack[i][masks_arg_stack[i]] = self.img_d[tuple(masks_arg[i].T)]
            if flips[i]:
                self.img_stack[i] = np.flip(self.img_stack[i], axis=1)
        
        return self.img_stack


# masks from SAM model have overlap with eachother, combinging them and then finding contours fix the overlap issue
def contours_fromMask(masks):
    mask_union = False
    for mask in masks:
        mask_union = mask_union | mask['segmentation']
    mask_union = mask_union.astype('uint8')
    contours, _ = cv2.findContours(mask_union, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_union


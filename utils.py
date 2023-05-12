
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tifffile as tiff
from Image import Image


GLOBAL_VARS = {
    'img':[],
}


class ImageDisp(tk.Frame):
    canv_width = 777
    canv_height = 450
    
    def __init__(self, master):
        tk.Frame.__init__(self, master, width=800, height=560, padx=10, pady=10)
        self.grid_propagate(False)
        
        self.canv = tk.Canvas(self, width=ImageDisp.canv_width, height=ImageDisp.canv_height, bg='white')        
        self.btn_load = tk.Button(self, text="Load Data", command=self.load_image)        
        self.frame_obj = tk.Frame(self, width=80, height=20, padx=10, pady=2)
        # self.obj_det = ObjectDetector(self)                
        self.lbl_point = tk.Label(self.frame_obj, text="Points per side")
        self.entry_point = tk.Entry(self.frame_obj, bd=1.5, width=3, justify='center')
        self.entry_point.insert(0, '16')
        self.lbl_iou = tk.Label(self.frame_obj, text="IoU threshold")
        self.entry_iou = tk.Entry(self.frame_obj, bd=1.5, width=3, justify='center')
        self.entry_iou.insert(0, '.9')
        self.lbl_size = tk.Label(self.frame_obj, text="Min size")
        self.entry_size = tk.Entry(self.frame_obj, bd=1.5, width=3, justify='center')
        self.entry_size.insert(0, '35')
        self.btn_det = tk.Button(self.frame_obj, text="Detect Objects", command=self.seg_image)        
        self.btn_rem = tk.Button(self.frame_obj, text="Remove Objects", command=self.rem_object)
        self.btn_comb = tk.Button(self.frame_obj, text="Combine Objects", command=self.com_object)
        self.btn_sort = tk.Button(self.frame_obj, text="Sort Objects", command=self.sort_object)
        self.btn_flip = tk.Button(self.frame_obj, text="Flip Objects")               
        self.btn_stack = tk.Button(self, text="Convert to Stack", command=self.to_stack)        
        
        self.canv.grid(row=0, column=0, columnspan=3)
        self.btn_load.grid(row=1, column=0, padx= 5, pady=5)        
        self.frame_obj.grid(row=1, column=1, padx= 5, pady=5)                      
        self.lbl_point.place(x=5, y=0)
        self.entry_point.place(x=90, y=0)
        self.lbl_iou.place(x=125, y=0)
        self.entry_iou.place(x=205, y=0)
        self.lbl_size.place(x=240, y=0)
        self.entry_size.grid(row=0, column=2, padx= 5, pady=0, sticky='E')        
        self.btn_det.grid(row=1, column=0, padx= 5, pady=2)        
        self.btn_rem.grid(row=1, column=1, padx= 5, pady=2)
        self.btn_comb.grid(row=1, column=2, padx=5, pady=2)
        self.btn_sort.place(x=60, y=53)
        self.btn_flip.grid(row=2, column=1, columnspan=2, padx= 60, pady=2)               
        self.btn_stack.grid(row=1, column=2, padx= 5, pady=5)        

    
    def load_image(self):                   
        self.file_path = tk.filedialog.askopenfilename(title = "select image",
                                                       filetypes = (("ndpi images", "*.ndpi"),
                                                                    ("tiff images", "*.tif"),
                                                                    ("jpeg images", "*.jpg")))
        GLOBAL_VARS['img'] = Image(tiff.imread(self.file_path), (ImageDisp.canv_width, ImageDisp.canv_height))
        self.canv.create_image(ImageDisp.canv_width/2,
                                ImageDisp.canv_height/2,
                                image=GLOBAL_VARS['img'].img_pil)
    
    
    def seg_image(self):
        points_per_side = int(self.entry_point.get())
        pred_iou_thresh=float(self.entry_iou.get())
        min_contour_len=float(self.entry_size.get())
        contours = GLOBAL_VARS['img'].segment(points_per_side, pred_iou_thresh, min_contour_len)
        for con in contours:
            self.canv.create_line(con.flatten().tolist(), width=2, fill='green', tag='border')

    def rem_object(self):        
        def callback(event): 
            con = GLOBAL_VARS['img'].remove_object(event)
            self.canv.create_line(con.flatten().tolist(), width=2, fill='red', tag='border')
        self.canv.bind("<Button-1>", callback)

    def com_object(self):
        sqr = {'flag':0, 'x0':0, 'y0':0, 'x1':0, 'y1':0}
        def callback_motion(event):
            if sqr['flag'] == 0:
                sqr['x0'] = event.x
                sqr['y0'] = event.y
                sqr['flag'] = 1
            else:
                sqr['x1'] = event.x
                sqr['y1'] = event.y
                self.canv.delete('square')
                self.canv.create_rectangle(sqr['x0'], sqr['y0'], sqr['x1'], sqr['y1'], outline="#fb0", tag='square')
        
        def callback_release(event):
            self.canv.delete('square')
            if sqr['x0'] > sqr['x1']:
                sqr['x0'], sqr['x1'] = sqr['x1'], sqr['x0']
            if sqr['y0'] > sqr['y1']:
                sqr['y0'], sqr['y1'] = sqr['y1'], sqr['y0']
            sqr['flag'] = 0            
            contours = GLOBAL_VARS['img'].combine_objects(sqr)
            for con in contours:
                self.canv.create_line(con.flatten().tolist(), width=2, fill='#fb0', tag='border')
        
        self.canv.unbind("<Button-1>")
        self.canv.bind("<B1-Motion>", callback_motion)
        self.canv.bind("<ButtonRelease-1>", callback_release)
        
    def sort_object(self):
        self.canv.unbind("<B1-Motion>")
        self.canv.unbind("<ButtonRelease-1>")
        self.canv.delete('border')
        contours, contours_centroid = GLOBAL_VARS['img'].sort_objects()
        for con_group in contours:
            for con in con_group:
                self.canv.create_line(con.flatten().tolist(), width=2, fill='green')
            
        self.ents = multiEntry(self.canv, len(contours), bd=1, width=5)
        self.ents.place(contours_centroid[:,1], contours_centroid[:,0])
        self.ents.insert(0, np.arange(len(contours)))
        
    def to_stack(self):        
        idx_slices = np.argsort(self.ents.get())
        stack = GLOBAL_VARS['img'].convert_to_stack(idx_slices)
        stack_path = self.file_path.replace('.tif', '_stack.tif')       
        tiff.imwrite(stack_path, stack)
        


class multiEntry:
    def __init__(self, master, no, **kwargs):
        self.no = no
        self.ents = [tk.Entry(master, **kwargs) for _ in range(no)]
        
    def place(self, posx, posy):
        for (ent, px, py) in zip(self.ents, posx, posy):
            ent.place(x=px, y=py)
            
    def insert(self, index, numbers):
        for (ent, num) in zip(self.ents, numbers):
            ent.insert(index, str(num))     
            
    def get(self):
        numbers = [int(ent.get()) for ent in self.ents]
        return numbers


















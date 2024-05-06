from skimage.feature import hog


class HOGFeatureExtractor:
    def __init__(self):
        """
        :param network_type: network constructor function name (string). See torchvision.models.resnet.__all__
        :param pretrained:
        """
        super().__init__()
        # define each block as 4x4 cells of 64x64 pixels each
        self.cell_size = (8, 8)      # h x w in pixels
        self.block_size = (1, 1)         # h x w in cells
        self.num_orientations = 9  # number of orientation bins
        self.img_size = (0, 0)
        self.n_cells = (0, 0)
        # create a HOG object
           
    def forward(self, img):
        self.img_size = img.shape[:2] 
        self.n_cells = (self.img_size[0] // self.cell_size[0], self.img_size[1] // self.cell_size[1])
        hog_feats = hog(img,
                    orientations=self.num_orientations,
                    pixels_per_cell=(self.cell_size[0], self.cell_size[1]),
                    cells_per_block=(self.block_size[0], self.block_size[1]),
                    multichannel=True
                )   
        # reshape into spatial hierarchy of windows, blocks, cells, and histogram bins. 
        # For example, hog_feats[i][j] corresponds to the window (in numpy slicing syntax)
        hog_feats_width = self.img_size[0] // self.cell_size[0]
        hog_feats_height = self.img_size[1] // self.cell_size[1]
        hog_feats = hog_feats.reshape(hog_feats_width,
                                      hog_feats_height,
                                      self.num_orientations) # should we reshape in this way?
        return hog_feats
       

import cv2
 
# Load the image and convert to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# define each block as 4x4 cells of 64x64 pixels each
cell_size = (128, 128)      # h x w in pixels
block_size = (4, 4)         # h x w in cells
win_size = (8, 6)           # h x w in cells
 
nbins = 9  # number of orientation bins
img_size = img.shape[:2]  # h x w in pixels
 
# create a HOG object
hog = cv2.HOGDescriptor(
    _winSize=(win_size[1] * cell_size[1],
              win_size[0] * cell_size[0]),
    _blockSize=(block_size[1] * cell_size[1],
                block_size[0] * cell_size[0]),
    _blockStride=(cell_size[1], cell_size[0]),
    _cellSize=(cell_size[1], cell_size[0]),
    _nbins=nbins
)
n_cells = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])
 
# find features as a 1xN vector, then reshape into spatial hierarchy
hog_feats = hog.compute(img)
hog_feats = hog_feats.reshape(
    n_cells[1] - win_size[1] + 1,
    n_cells[0] - win_size[0] + 1,
    win_size[1] - block_size[1] + 1,
    win_size[0] - block_size[0] + 1,
    block_size[1],
    block_size[0],
    nbins)
print(hog_feats.shape)

class HOGFeatureExtractor:
    def __init__(self):
        """
        :param network_type: network constructor function name (string). See torchvision.models.resnet.__all__
        :param pretrained:
        """
        super().__init__()
        # define each block as 4x4 cells of 64x64 pixels each
        self.cell_size = (128, 128)      # h x w in pixels
        self.block_size = (4, 4)         # h x w in cells
        self.win_size = (8, 6)           # h x w in cells
        self.nbins = 9  # number of orientation bins
        self.img_size = (0, 0)
        self.n_cells = (0, 0)
        # create a HOG object
        self.hog = cv2.HOGDescriptor(
            _winSize=(self.win_size[1] * self.cell_size[1],
                    self.win_size[0] * self.cell_size[0]),
            _blockSize=(self.block_size[1] * self.cell_size[1],
                        self.block_size[0] * self.cell_size[0]),
            _blockStride=(self.cell_size[1], self.cell_size[0]),
            _cellSize=(self.cell_size[1], self.cell_size[0]),
            _nbins=self.nbins
        )      

    def forward(self, img):
        self.img_size = img.shape[:2] 
        self.n_cells = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])
        hog_feats = hog.compute(img)
        # reshape into spatial hierarchy of windows, blocks, cells, and histogram bins. 
        # For example, hog_feats[i][j] corresponds to the window (in numpy slicing syntax)
        hog_feats = hog_feats.reshape(
            self.n_cells[1] - self.win_size[1] + 1,
            self.n_cells[0] - self.win_size[0] + 1,
            self.win_size[1] - self.block_size[1] + 1,
            self.win_size[0] - self.block_size[0] + 1,
            self.block_size[1],
            self.block_size[0],
            self.nbins) # should we reshape in this way?
        return hog_feats
       
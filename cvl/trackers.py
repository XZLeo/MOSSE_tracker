import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch, crop_patch2
from copy import copy
import matplotlib.pyplot as plt
from skimage.transform import resize
from cvl.features_resnet import DeepFeatureExtractor
from cvl.features_HOG import HOGFeatureExtractor

class NCCTracker:

    def __init__(self, learning_rate=0.007):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def get_region(self):
        return copy(self.region)

    def get_normalized_patch(self, image):
        region = self.region
        patch = crop_patch(image, region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.get_normalized_patch(image)
        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.get_normalized_patch(image)
        
        patchf = fft2(patch)

        responsef = self.template * np.conj(patchf)
        response = ifft2(responsef).real

        r, c = np.unravel_index(np.argmax(response), response.shape)
        print("/n")
        print("row on patch coordinate: ", r)
        print("column on patch coordinate: ", c)
        print("/n")

        # Keep for visualisation
        self.last_response = response

        # r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        # c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        r_offset = int(r - self.region.height/2)
        c_offset = int(c - self.region.width/2)
        print("row offset: ", r_offset)
        print("column offset: ", c_offset)
        print("/n")

        # self.region.xpos -= c_offset
        # self.region.ypos -= r_offset

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        # self.region.xpos -= c
        # self.region.ypos -= r

        return self.get_region()

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.get_normalized_patch(image)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr

class MoSSETracker:

    def __init__(self, learning_rate=0.05):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

        self.learned_filters = []
        self.A_ts = []
        self.B_ts = []
        self.responses = []

        self.sigma = 1.5

        self.num_channels = None

    def get_region(self):
        return copy(self.region)

    def get_normalized_patch(self, image):
        region = self.region
        patch = crop_patch(image, region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch
    
    def _gaussian2(self, x: float, y: float, sigma: float):
        return np.exp(-(x**2 + y**2)/(2 * sigma**2))

    def _get_G(self):
        target_score_map = np.array([
            [self._gaussian2(col - self.region_center[1], 
                            row - self.region_center[0], self.sigma) 
                            for col in range(self.region.width)]
            for row in range(self.region.height)])
        
        return fft2(target_score_map)


    def start(self, image, region):
        """_summary_

        Args:
            image (_type_): RGB image
            region (_type_): bounding boxes
        """
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"

        self.num_channels = image.shape[2]  # Number of input channels

        for channel in range (self.num_channels):
            self.region = copy(region)

            self.region_shape = (region.height, region.width)
            self.region_center = (region.height // 2, region.width // 2)
            patch = self.get_normalized_patch(image[:,:,channel])    # Crop and then Normalize at one channel
            
            self.template = fft2(patch)
            F = self.template                                        # Compute F for each channel
            G = self._get_G()                                        # Create Gaussian distribution for each channel
            A_t = G * np.conj(F)
            B_t = F * np.conj(F) 
            learned_filter = A_t / B_t                   # Compute filter H* for each channels
            self.learned_filters.append(learned_filter)             # Store initialized filrer for all channels
            self.A_ts.append(A_t)                                   # Store A0 and B0 for all channels
            self.B_ts.append(B_t)

    def detect(self, image):
        for channel in range(self.num_channels):
            
            patch = self.get_normalized_patch(image[:,:,channel])   # Get patch for one channel
            
            patchf = fft2(patch)

            responsef = patchf * (self.learned_filters[channel])    # Compute resonsef for each channel
            response = ifft2(responsef).real
            self.responses.append(response)
        
        res = np.sum(self.responses, axis=0)                        # Compute sum of response along each channel
                                                                    # Axis = 0 because self.responses is a list of response

        r, c = np.unravel_index(np.argmax(res), res.shape)

        # Keep for visualisation
        self.last_response = res

        r_offset = round(r - self.region.height//2)
        c_offset = round(c - self.region.width//2)

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.get_region()

    def update(self, image, lr=0.1):
        for channel in range (self.num_channels):
            patch = self.get_normalized_patch(image[:,:,channel])
            
            patchf = fft2(patch)
            F = patchf
            G = self._get_G()
            A_t = self.learning_rate*G * np.conj(F) + (1-self.learning_rate)* self.A_ts[channel]
            B_t = self.learning_rate*F * np.conj(F) + (1-self.learning_rate)* self.B_ts[channel]  
            learned_filter = A_t / B_t

            # Update filter and At, Bt for each channel
            self.learned_filters[channel] = learned_filter
            self.A_ts[channel] = A_t
            self.B_ts[channel] = B_t


class MoSSETrackerDeepFeature:

    def __init__(self, learning_rate=0):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

        self.learned_filters = []
        self.A_ts = []
        self.B_ts = []
        self.responses = []

        self.sigma = 1.5

        self.num_channels = None

        self.image_channels = None
        # Initiazline Deep Extrctor
        self.feature_extractor = DeepFeatureExtractor()
        self.feature_size = None
        self.patch_deep_features = None

    def get_region(self):
        return copy(self.region)

    def get_normalized_patch(self, image):
        region = self.region
        patch = crop_patch(image, region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch
    
    def _gaussian2(self, x: float, y: float, sigma: float):
        return np.exp(-(x**2 + y**2)/(2 * sigma**2))

    def _get_G_deep_feature(self, feature_size):
        width = feature_size[1]
        height = feature_size[0]
        cen_col = width // 2
        cen_row = height // 2
        target_score_map = np.array([
            [self._gaussian2(col - cen_col, 
                            row - cen_row, self.sigma) 
                            for col in range(width)]
            for row in range(height)])
        return fft2(target_score_map)
    
    def _convert_tensor_numpy(self, tensor_feature):
        tensor_cpu = tensor_feature.cpu()
        array = tensor_cpu.numpy()
        return array
    
    def _extract_deep_feature(self, img_patch):
        """ Input: image
            Output: array converted from feature tensors (W,H,C)"""
        img_patch = resize(img_patch, (224, 224))

        # Extract deep features (x1, x2, x3, x4)
        path_feature_x1, path_feature_x2, path_feature_x3, path_feature_x4 = self.feature_extractor.forward(img_patch)
        
        # SELECT DEEP FEATURE
        patch_feature_tensor = path_feature_x2
        # Convert tensor to numpy array
        patch_feature_numpy = self._convert_tensor_numpy(patch_feature_tensor)

        patch_feature_numpy = np.squeeze(patch_feature_numpy, axis=0)
        patches = np.reshape(patch_feature_numpy, (patch_feature_numpy.shape[1], patch_feature_numpy.shape[2], -1))    # Convert from (1,channels,w,h) to (w,h,channels)

        # number of deep feature channels and feature size
        self.num_channels = patches.shape[2]
        print('number of deep feature channels: %d' %self.num_channels)
        self.feature_size = (patches.shape[0], patches.shape[1])
        return patches

    def start(self, image, region):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"

        # self.num_channels = image.shape[2]  # Number of input channels
        # CROP REGION OUT OF IMAGE
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)   

        self.image_channels = image.shape[2]

        region = self.region
        img_patch = crop_patch2(image, region, self.image_channels)
        self.patch_deep_features = self._extract_deep_feature(img_patch)
        patches = self.patch_deep_features

        for channel in range (self.num_channels):               
            
            self.template = fft2(patches[:,:,channel])               # Transform each channel to Fourier domain
            F = self.template                                        # Compute F for each channel
            G = self._get_G_deep_feature(self.feature_size)               # Create Gaussian distribution for each channel
            A_t = G * np.conj(F)
            B_t = F * np.conj(F) 
            learned_filter = A_t / B_t                   # Compute filter H* for each channels
            self.learned_filters.append(learned_filter)             # Store initialized filrer for all channels
            self.A_ts.append(A_t)                                   # Store A0 and B0 for all channels
            self.B_ts.append(B_t)

    def detect(self, image):
        region = self.region
        img_patch = crop_patch2(image, region, self.image_channels)
        self.patch_deep_features = self._extract_deep_feature(img_patch)

        patches = self.patch_deep_features

        for channel in range(self.num_channels):
            
            # patch = self.get_normalized_patch()   # Get patch for one channel
            
            patchf = fft2(patches[:,:,channel])

            responsef = patchf * (self.learned_filters[channel])    # Compute responsef for each channel
            response = ifft2(responsef).real
            self.responses.append(response)
        
        res = np.sum(self.responses, axis=0)                        # Compute sum of response along each channel
                                                                    # Axis = 0 because self.responses is a list of response
        res = resize(res, (self.region.height, self.region.width))

        r, c = np.unravel_index(np.argmax(res), res.shape)

            # Keep for visualisation
        self.last_response = res

        r_offset = round(r - self.region.height/2)
        c_offset = round(c - self.region.width/2)

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.get_region()

    def update(self, image, lr=0.1):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        # patch = self.get_normalized_patch(image)
        # patchf = fft2(patch)
        # self.template = self.template * (1 - lr) + patchf * lr

        patches = self.patch_deep_features

        for channel in range (self.num_channels):
            
            patchf = fft2(patches[:,:, channel])
            F = patchf
            G = self._get_G_deep_feature(self.feature_size)
            A_t = self.learning_rate*G * np.conj(F) + (1-self.learning_rate)* self.A_ts[channel]
            B_t = self.learning_rate*F * np.conj(F) + (1-self.learning_rate)* self.B_ts[channel]  
            learned_filter = A_t / B_t

            # Update filter and At, Bt for each channel
            self.learned_filters[channel] = learned_filter
            self.A_ts[channel] = A_t
            self.B_ts[channel] = B_t
            
            
class MoSSETrackerManual(MoSSETracker):
    def __init__(self, learning_rate=0.1, filter_type:str='HOG'):
        super().__init__(learning_rate)
        self.feature_extractor = HOGFeatureExtractor()
        
    def start(self, image, region):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"

        # self.num_channels = image.shape[2]  # Number of input channels
        # CROP REGION OUT OF IMAGE
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)   

        self.image_channels = image.shape[2]

        region = self.region
        img_patch = crop_patch2(image, region, self.image_channels)
        self.patch_deep_features = self._extract_manual_feature(img_patch)
        patches = self.patch_deep_features
        
        for channel in range (self.num_channels):                
            self.template = fft2(patches[:,:,channel])               # Transform each channel to Fourier domain
            F = self.template                                        # Compute F for each channel
            G = self._get_G_feature(self.feature_size)               # Create Gaussian distribution for each channel???
            A_t = G * np.conj(F)
            B_t = F * np.conj(F) 
            learned_filter = A_t / B_t                   # Compute filter H* for each channels
            self.learned_filters.append(learned_filter)             # Store initialized filrer for all channels
            self.A_ts.append(A_t)                                   # Store A0 and B0 for all channels
            self.B_ts.append(B_t)
        return
    
    def _get_G_feature(self, feature_size):
        width = feature_size[1]
        height = feature_size[0]
        cen_col = width // 2
        cen_row = height // 2
        target_score_map = np.array([
            [self._gaussian2(col - cen_col, 
                            row - cen_row, self.sigma) 
                            for col in range(width)]
            for row in range(height)])
        # plt.imshow(target_score_map)
        # plt.show()
        return fft2(target_score_map)
        
    def _extract_manual_feature(self, img_patch):
        """ Input: image
            Output: array converted from feature tensors (W,H,C)"""
        img_patch = resize(img_patch, (224, 224))

        # Extract features from manually designed filters
        manual_features = self.feature_extractor.forward(img_patch)

        # number of deep feature channels and feature size
        self.num_channels = manual_features.shape[2] #? reshape accordingly!!
        # print('number of deep feature channels: %d' %self.num_channels)
        self.feature_size = (manual_features.shape[0], manual_features.shape[1])
        return manual_features
    
    def detect(self, image):
        region = self.region
        img_patch = crop_patch2(image, region, self.image_channels)
        self.patch_deep_features = self._extract_manual_feature(img_patch)

        patches = self.patch_deep_features

        for channel in range(self.num_channels):
            
            # patch = self.get_normalized_patch()   # Get patch for one channel
            
            patchf = fft2(patches[:,:,channel])

            responsef = patchf * (self.learned_filters[channel])    # Compute responsef for each channel
            response = ifft2(responsef).real
            self.responses.append(response)
        
        res = np.sum(self.responses, axis=0)                        # Compute sum of response along each channel
                                                                    # Axis = 0 because self.responses is a list of response
        res = resize(res, (self.region.height, self.region.width))

        r, c = np.unravel_index(np.argmax(res), res.shape)

            # Keep for visualisation
        self.last_response = res

        r_offset = round(r - self.region.height/2)
        c_offset = round(c - self.region.width/2)

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.get_region()
    
    def update(self):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        # patch = self.get_normalized_patch(image)
        # patchf = fft2(patch)
        # self.template = self.template * (1 - lr) + patchf * lr

        patches = self.patch_deep_features

        for channel in range (self.num_channels):
            
            patchf = fft2(patches[:,:, channel])
            F = patchf
            G = self._get_G_feature(self.feature_size)
            A_t = self.learning_rate*G * np.conj(F) + (1-self.learning_rate)* self.A_ts[channel]
            B_t = self.learning_rate*F * np.conj(F) + (1-self.learning_rate)* self.B_ts[channel]  
            learned_filter = A_t / B_t

            # Update filter and At, Bt for each channel
            self.learned_filters[channel] = learned_filter
            self.A_ts[channel] = A_t
            self.B_ts[channel] = B_t
        return
    
    
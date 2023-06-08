"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from models.RecursiveUNet import RUNet
from models.UNet import UNET
from utils.utils import med_reshape


class InferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNET(nclass=3,aux=False, in_channels=1)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first
        Arguments:
            volume {Numpy array} -- 3D array representing the volume
        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size
        Arguments:
            volume {Numpy array} -- 3D array representing the volume
        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        with torch.no_grad():
            for idx0 in range(volume.shape[0]):
                slc = volume[idx0, :, :]
                slc = slc[None, None, :]
                slc_ts = torch.from_numpy(slc)
                slc_ts = slc_ts.to(self.device, dtype=torch.float)

                prediction = self.model(slc_ts)
                
                msk = prediction.argmax(axis=1).cpu().numpy()
                slices += [msk]
                
        return np.concatenate(slices, axis=0)
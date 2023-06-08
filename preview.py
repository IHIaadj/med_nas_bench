import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

test_load = np.array(nib.load("Task07_Pancreas-007\Task07_Pancreas\imagesTr\pancreas_077.nii.gz").dataobj)
print(test_load)
print(test_load.shape)

plt.imshow(test_load[:, :,5], cmap='bone')
plt.axis('off')
plt.show()
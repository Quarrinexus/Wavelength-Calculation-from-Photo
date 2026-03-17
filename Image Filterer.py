from skimage import io, img_as_ubyte
import numpy as np

single_slit = True # whether this is a single slit dataset (True) or double slit (False)

def apply_filter(img, n, threshold=0.5, threshold_max=1):
    # Store original shape for reshaping later
    original_shape = img.shape

    filtered_img = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] < threshold or img[i][j] > threshold_max:
                filtered_img.append(0)
            else:
                filtered_img.append(img[i][j])

    img = np.array(filtered_img).reshape(original_shape)

    output = img_as_ubyte(img)
    if single_slit:
        io.imsave(f'Single Slit/Filtered_Images/Filtered_Image_{n}.jpeg', output)
    else:
        io.imsave(f'Filtered_Images/Filtered_Image_{n}.jpeg', output)

# Loop through images and apply filter
for n in range(1, 14):
    if single_slit:
        img = io.imread(f'Single Slit/Images/Image_{n}.jpeg', as_gray=True)
    else:
        img = io.imread(f'Images/Image_{n}.jpeg', as_gray=True)
    apply_filter(img, n)
    print(f"Filtered Image {n} saved.")
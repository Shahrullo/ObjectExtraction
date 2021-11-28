import cv2
import numpy as np
import argparse

from utils import *


# define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to the input image', required=True)
parser.add_argument('-n', '--new-background', dest='new_background', action='store_true')
args = vars(parser.parse_args())

# read the image and convert to binary
image = cv2.imread(args['input'])
show('Input image', image)
# blur the image to smmooth out the edges a bit, also reduces a bit of noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# convert the image to grayscale
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
# apply thresholding to conver the image to binary format
# after this operation all the pixels below 200 value will be 0...
# and all th pixels above 200 will be 255
ret, gray = cv2.threshold(gray, 200 , 255, cv2.CHAIN_APPROX_NONE)


# find the largest contour area in the image
contour = find_largest_contour(gray)
image_contour = np.copy(image)
cv2.drawContours(image_contour, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
show('Contour', image_contour)


# create a black `mask` the same size as the original grayscale image
mask = np.zeros_like(gray)
# fill the new mask with the shape of the largest contour
# all the pixels inside that area will be white
cv2.fillPoly(mask, [contour], 255)
# create a copy of the current mask
res_mask = np.copy(mask)
res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels


# create a mask for obvious and probable foreground pixels
# al the obvious foreground pixels will be white and probable ones will be black
mask2 = np.where((res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')


# create 'new_mask3d' from 'mask2' but with 3 dimensions
new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
mask3d = new_mask3d
mask3d[new_mask3d > 0] = 255.0
mask3d[mask3d > 255] = 255.0
# apply Gaussian blurring to smoothen out the edges a bit
# mask3d is the final foreground mask (not extracted foreground)
mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)
show('Foreground mask', mask3d)


# create the foreground image by zeroing out the pixels where 'mask2' has black pixels
foreground = np.copy(image).astype(float)
foreground[mask2 == 0] = 0
show('Foreground', foreground.astype(np.uint8))


# save images to disk
save_name = args['input'].split('/')[-1].split('.')[0]
cv2.imwrite(f'outputs/{save_name}_foreground.png', foreground)
cv2.imwrite(f'outputs/{save_name}_foreground_mask.png', mask3d)
cv2.imwrite(f'outputs/{save_name}_contour.png', image_contour)


# the '--new_background' flag is 'True', then apply the new background
if args['new_background']:
    apply_new_background(mask3d, foreground, save_name)



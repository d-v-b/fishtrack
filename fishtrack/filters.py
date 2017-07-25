from .util import long_tail_threshold, get_bbox
from .measurements import remotest_point

def get_fish_mask(im, sigma=25, shrink_factors=[.05,.1], min_sizes=[250,250], disk_size=5, return_sparse=True):
    """
    Get a binary mask representing the fish nervous system from a fluorescence image of a fish.
    :param im:
    :param sigma:
    :param shrink_factors:
    :param min_sizes:
    :param disk_size:
    :param return_sparse:
    :return:
    """

    from skimage.morphology import remove_small_objects, disk, binary_opening
    from scipy.sparse import coo_matrix
    
    im_ = im.copy().astype('float32')

    # remove background
    im_bgr = remove_background(im, sigma=sigma)

    # estimate a threshold on the background-removed image
    thr = long_tail_threshold(im_bgr, shrink_factor=shrink_factors[0])

    mask = remove_small_objects(im_bgr > thr, min_size=min_sizes[0])
    mask = disk_dilate(mask, disk_size=disk_size)
    
    # perform secondary thresholding on stuff in the mask
    thr_2 = long_tail_threshold(im_[mask], shrink_factor=shrink_factors[1])
    mask = mask * (im_ > thr_2)
    
    # remove small objects again
    mask = remove_small_objects(mask, min_size=min_sizes[1])

    if return_sparse:
        mask = coo_matrix(mask)

    return mask


# todo: remove this function, since it's not really necessary once you know the brain position in each image
def get_brain(im, return_sparse=True, brain_size=40):
    """
    Crop the input image around the brain
    :param im:
    :param return_sparse:
    :param brain_size:
    :return:
    """
    from scipy.sparse import issparse
    from numpy import array, zeros
    from scipy.sparse import coo_matrix

    im_ = im.copy().astype('bool')
    brain = zeros(im_.shape, dtype='bool')

    # if the input is sparse, make it dense
    if issparse(im):
        im_ = array(im_.todense())

    if not im_.any():
        if return_sparse:
            from scipy.sparse import coo_matrix
            brain = coo_matrix(brain)
        return brain

    # get the center of the brain
    y_, x_ = remotest_point(im_)
    crop = (slice(y_ - brain_size // 2, y_ + brain_size // 2), slice(x_ - brain_size // 2, x_ + brain_size // 2))
    brain[crop] = im_[crop]

    if return_sparse:
        brain = coo_matrix(brain)

    return brain


def remove_background(im, sigma=4):
    """
    Remove the background of an image by subtracting a gaussian-blurred version of itself.
    :param im:
    :param sigma:
    :return:
    """
    from scipy.ndimage.filters import gaussian_filter
    im_ = im.copy().astype('float32')
    return im_ - gaussian_filter(im_, sigma=sigma)


def disk_dilate(im, disk_size=8):
    """
    Wrapper for skimage.morphology.binary_dilation function optimized for dilating a sparse image. The morphological
    operation is only applied to a restricted region in the input image
    :param im:
    :param disk_size:
    :return:
    """
    from skimage.morphology import disk, binary_dilation
    # execution time of binary dilation depends on size of the image so we can speed things up by cropping the image
    rmin, rmax, cmin, cmax = get_bbox(im, pad=disk_size//2)
    result = im.copy()
    result[rmin:rmax, cmin:cmax] = binary_dilation(im[rmin:rmax, cmin:cmax], disk(disk_size))
    return result

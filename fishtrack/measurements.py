from .util import get_bbox

def tail_angle(images, center, radius):
    """
    Get estimate of tail angle from 3D numpy array of images of fish

    :param images:
    :param center:
    :param radius:
    :return:
    """
    from skimage.draw import circle_perimeter
    from numpy import arctan2, pi, argmax, unwrap, argsort

    cent_y, cent_x = center

    y_c, x_c = circle_perimeter(c=cent_x, r=cent_y, radius=radius, shape=images.shape[1:])
    
    angle_range = arctan2(y_c - cent_y, -(x_c - cent_x))
    inds = argsort(angle_range)
    keep = inds
    x_tail = x_c[keep]
    y_tail = y_c[keep]
    tail_arc = images[:, y_tail, x_tail]
    
    angle = argmax(tail_arc, axis=1)
    #angle = unwrap(angle_range[keep][angle])
    angle = angle_range[keep][angle]
    angle = angle.astype('float32')

    return angle, (x_tail, y_tail), angle_range[keep]


def remotest_point(mask):
    """
    Get the (row, column) coordinates of the point in a binary image with the greatest distance from the border
    :param mask:
    :return:
    """
    from scipy.ndimage.measurements import maximum_position
    from scipy.ndimage.morphology import distance_transform_edt
    from scipy.sparse import issparse
    from numpy import array

    dist_tx = distance_transform_edt

    mask_ = mask.copy().astype('float')

    if issparse(mask_):
        mask_ = array(mask_.todense())

    # without some padding, the bounding box can be too close to the edge of the mask
    rmin, rmax, cmin, cmax = get_bbox(mask_, pad=20)
    mask_[rmin:rmax, cmin:cmax] = dist_tx(mask_[rmin:rmax, cmin:cmax])
    result = maximum_position(mask_)

    return result


def eigenvalues(im):
    """
    Return eigenvalues and eigenvectors, sorted by eigenvalue
    Wrapper for scipy.linalg.eig()
    :param im:
    :return:
    """
    from numpy import where, cov, argsort
    from scipy.linalg import eig

    y, x = where(im)
    evals, evecs = eig(cov(x, y))
    idx = argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    return evals, evecs


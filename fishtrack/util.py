
def long_tail_threshold(data, shrink_factor=.1, nbins=512):
    from numpy import histogram, where, median, argmin, abs
    counts, bins = histogram(data.ravel(), bins=nbins)
    pk = median(data)
    pk_ind = argmin(abs(bins - pk))
    pk_counts = counts[pk_ind-1]
    right_tail = where(counts[pk_ind:] < (pk_counts * shrink_factor))[0]
    if len(right_tail) > 0:
        right_foot = right_tail[0] + pk_ind
    else:
        right_foot = -1
    return bins[right_foot]


# get the bounding box of a binary array with some padding
def get_bbox(img, pad=0):
    from scipy.sparse import issparse
    from numpy import any, where, array

    img_ = img.copy()

    if issparse(img_):
        img_ = array(img_.todense())
        
    rmin, cmin = 0,0
    rmax, cmax = array(img_.shape) - 1
    
    rows = any(img_, axis=1)
    cols = any(img_, axis=0)
    
    if rows.any():
        rmin, rmax = where(rows)[0][[0, -1]]
        cmin, cmax = where(cols)[0][[0, -1]]

    return rmin - pad, rmax + pad, cmin - pad, cmax + pad


# convert from cartesian to polar coordinates
def cart2pol(x, y):
    from numpy import hypot, arctan2
    r = hypot(x, y)
    phi = arctan2(y, x)
    return r, phi


# make a 2d rotation matrix
def rotmat2d(phi):
    from numpy import cos, sin, zeros
    mat = zeros([2, 2])
    mat[0, 0] = cos(phi)
    mat[1, 1] = cos(phi)
    mat[0, 1] = sin(phi)
    mat[1, 0] = -sin(phi)
    return mat


# compute the difference between two angles
def angle_difference(x, y):
    from numpy import arctan2, sin, cos
    return arctan2(sin(x-y), cos(x-y))

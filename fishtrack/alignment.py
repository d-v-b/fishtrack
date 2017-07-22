from .filters import get_brain, get_fish_mask
from .util import long_tail_threshold, cart2pol, angle_difference
from .measurements import eigenvalues, remotest_point


def align_brains(static, moving, brain_position):
    from dipy.align.imaffine import AffineRegistration, AffineMap
    from dipy.align.transforms import RigidTransform2D 
    from scipy.ndimage.measurements import center_of_mass
    from numpy import round, eye, array

    affreg = AffineRegistration(verbosity=0)
    rigid = RigidTransform2D()

    static_, moving_ = static.copy(), moving.copy()

    static_brain = get_brain(static_, return_sparse=False)
    moving_brain = get_brain(moving_, return_sparse=False)

    fill_value = static[static_brain].min()

    if not moving_brain.any():
        return moving_, AffineMap(eye(3))

    # isolate region of interest for registration
    w_y = slice(brain_position[0] -40, brain_position[0] + 40)
    w_x = slice(brain_position[1] -50, brain_position[1] + 50)
    static_brain_cropped = static_[w_y,w_x]
    moving_brain_cropped = moving_[w_y,w_x]

    # estimate rigid transformation between static and moving
    g2w = eye(3)
    g2w[:2, -1] = -array(static_brain_cropped.shape) / 2
    
    # first align centers of mass in the field of view
    params0 = None
    
    starting_affine = eye(3)
    tx = affreg.optimize(static_brain_cropped, moving_brain_cropped, rigid, params0, static_grid2world=g2w,
                         moving_grid2world=g2w, starting_affine=starting_affine)

    
    tx.domain_grid2world[:2, -1] = -brain_position
    warped = tx.transform(moving_, sampling_grid_shape=moving_.shape)
    
    return warped, tx


def get_cropped_fish(image, phi, dydx, crop_window):
    # given an image, a rotation angle, and a shift, apply the shift, then the rotation, then crop around the
    # center of the transformed image and apply a binary mask, returning an image of the fish nervous system on a
    # a background of 0s

    from skimage.transform import rotate
    from numpy import roll, round, array, rad2deg

    shifted = roll(image, round(dydx).astype('int'), axis=(0, 1))
    rotated = rotate(shifted, angle=rad2deg(phi), mode='wrap', preserve_range=True, order=3)

    mid = array(rotated.shape) // 2
    window_y, window_x = crop_window

    # we take the transpose to cancel the effect of this kind of indexing
    crop = rotated[mid[0] + window_y, mid[1] + window_x].T

    return crop
    


# point the tail of the fish toward the origin, parallel to the x-axis
# im must be a binarized fish mask
def orient_tail(im, brain_center, body_center, brain_size=20):
    from numpy import pi, array, abs, argmin, eye
    from scipy.sparse import issparse

    im_ = im.copy()

    if issparse(im_):
        im_ = array(im_.todense())

    tform = eye(3)
    dydx = (0,0)
    phi = 0
    y_, x_ = brain_center
    # assume the region in the image around the brain center is the brain
    brain = im_[y_-brain_size : y_ + brain_size, x_ - brain_size : x_ + brain_size]
    if not brain.any():
        return dydx, phi
        
    tail_y, tail_x = brain_center - body_center

    brain_tail_angle = cart2pol(tail_x, tail_y)[-1]

    # Eigenvector of largest eigenvalue corresponds to orientation of the long axis of the brain
    # unless we have eyes along with the brain, in which case we want the second largest eigenvalue
    evals, evecs = eigenvalues(brain)
    brain_angle = cart2pol(evecs[0, 0], evecs[1, 0])[-1]
    candidate_phis = array([brain_angle, brain_angle + pi])

    # Eigenvector of brain is ambiguous between tail to the left and tail to the right. Pick angle that
    # minimizes angular distance from brain-tail angle
    which_angle = argmin([abs(angle_difference(brain_tail_angle, candidate)) for candidate in candidate_phis])
    phi = candidate_phis[which_angle]

    # this vector translates the center of the brain to the center of the image
    dydx = (im_.shape[0] / 2) - y_, (im_.shape[1] / 2) - x_
    
    return dydx, phi
    

def centered_rotation(image, rotation_center, new_center, phi):
    """
    Return an affine matrix for rotating an image around a center point, then translating the center to a new point
    """
    from numpy import array, matrix
    from skimage.transform import warp, AffineTransform
    origin_y, origin_x = rotation_center
    shift_y, shift_x = new_center
    tf_rotate = AffineTransform(rotation=phi)
    tf_shift = AffineTransform(translation=[-origin_x, -origin_y])
    tf_shift_inv = AffineTransform(translation=[shift_x, shift_y])
    params = (tf_shift + (tf_rotate + tf_shift_inv)).params
    tform = matrix(params).I
            
    return tform

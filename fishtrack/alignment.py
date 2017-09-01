from .filters import get_brain, get_fish_mask
from .util import long_tail_threshold, cart2pol, angle_difference
from .measurements import eigenvalues, remotest_point


def align_brains(static, moving, brain_position, brain_size):
    from dipy.align.imaffine import AffineRegistration, AffineMap
    from dipy.align.transforms import RigidTransform2D, TranslationTransform2D
    from numpy import round, eye, array

    affreg = AffineRegistration(verbosity=0)
    translation = TranslationTransform2D()
    rigid = RigidTransform2D()

    # size of the brain, in pixels
    brain_h, brain_w = brain_size

    static_, moving_ = static.copy(), moving.copy()

    moving_brain = get_brain(moving_, return_sparse=False)

    if not moving_brain.any():
        return moving_, AffineMap(eye(3))

    # isolate region of interest for registration
    w_y = slice(brain_position[0] - brain_h//2, brain_position[0] + brain_h//2)
    w_x = slice(brain_position[1] - brain_w//2, brain_position[1] + brain_w//2)
    static_brain_cropped = static_[w_y, w_x]
    moving_brain_cropped = moving_[w_y, w_x]

    # estimate rigid transformation between static and moving
    g2w = eye(3)
    g2w[:2, -1] = -array(static_brain_cropped.shape) / 2
    
    # first align centers of mass in the field of view
    params0 = None
    
    starting_affine = eye(3)
    tx_translation = affreg.optimize(static_brain_cropped, moving_brain_cropped, translation, params0,
                                     static_grid2world=g2w, moving_grid2world=g2w, starting_affine=starting_affine)
    tx_rigid = affreg.optimize(static_brain_cropped, moving_brain_cropped, rigid, params0, static_grid2world=g2w,
                               moving_grid2world=g2w, starting_affine=tx_translation.affine)
    
    tx_rigid.domain_grid2world[:2, -1] = -brain_position
    warped = tx_rigid.transform(moving_, sampling_grid_shape=moving_.shape)
    
    return warped, tx_rigid


def get_cropped_fish(image, phi, brain_center, crop_window):
    """
    given an image, a rotation angle, and a point, rotate around the point then crop
    :param image:
    :param phi:
    :param brain_center:
    :param crop_window:
    :return:
    """

    from skimage.transform import rotate
    from numpy import roll, rad2deg

    brain_y, brain_x = brain_center
    # center the brain in the image so we don't index outside the bounds when cropping
    shifted = roll(image, (-brain_y + image.shape[0]//2, -brain_x + image.shape[1]//2), axis=(0, 1))
    rotated = rotate(shifted, angle=rad2deg(phi), mode='wrap', preserve_range=True, order=3,
                     center=(image.shape[1]//2, image.shape[0]//2))

    window_y, window_x = crop_window

    # we take the transpose to cancel side effects of this kind of indexing
    crop = rotated[image.shape[0]//2 + window_y, image.shape[1]//2 + window_x].T.astype(image.dtype)

    return crop
    

def orient_tail(im, brain_center, body_center, brain_size=20):
    """
    Given an masked fish image, a brain location, and a body location, find the angle of rotation that points that
    aligns the fish to the x-axis and points the head of the fish in the direction of increasing x-values

    :param im: binarized fish image
    :param brain_center: position of the fish brain
    :param body_center: position of the body
    :param brain_size: size, in pixels, of the fish brain
    :return: phi: the rotation angle needed to rotationally align the fish
    """
    from numpy import pi, array, abs, argmin, eye
    from scipy.sparse import issparse

    im_ = im.copy()

    if issparse(im_):
        im_ = array(im_.todense())

    phi = 0
    y_, x_ = brain_center
    # assume the region in the image around the brain center is the brain
    brain = im_[y_-brain_size: y_ + brain_size, x_ - brain_size: x_ + brain_size]
    if not brain.any():
        return phi
        
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
    #dydx = (im_.shape[0] / 2) - y_, (im_.shape[1] / 2) - x_
    
    return phi
    

def centered_rotation(rotation_center, new_center, phi):
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

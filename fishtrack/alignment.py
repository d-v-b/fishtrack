from .filters import get_brain, get_fish_mask
from .util import long_tail_threshold, cart2pol, angle_difference
from .measurements import eigenvalues, remotest_point


def align_brains(static, moving):
    from dipy.align.imaffine import AffineRegistration, AffineMap
    from dipy.align.transforms import RigidTransform2D
    from scipy.ndimage.measurements import center_of_mass
    from numpy import round, eye, array

    affreg = AffineRegistration(verbosity=0)
    rigid = RigidTransform2D()

    static_, moving_ = static.copy(), moving.copy()
    coms = (0, 0)

    static_brain = get_brain(static_, return_sparse=False)
    moving_brain = get_brain(moving_, return_sparse=False)

    fill_value = static[static_brain].min()

    if not moving_brain.any():
        return moving_, coms, AffineMap(eye(3))

    # isolate region of interest for registration
    coms = array(round(center_of_mass(static_brain)).astype('int'))
    static_brain_cropped = (static_brain * static)[coms[0] - 20:coms[0] + 20, coms[1] - 30:coms[1] + 30]
    moving_brain_cropped = (moving_brain * moving)[coms[0] - 20:coms[0] + 20, coms[1] - 30:coms[1] + 30]

    # set background values to minimum of brain to mitigate effect of masking on registration
    static_brain_cropped[static_brain_cropped == 0] = static_[static_brain].min()
    moving_brain_cropped[moving_brain_cropped == 0] = moving_[moving_brain].min()

    # estimate rigid transformation between static and moving
    g2w = eye(3)
    g2w[:2, -1] = -array(static_brain_cropped.shape) / 2

    params0 = None
    starting_affine = None
    tx = affreg.optimize(static_brain_cropped, moving_brain_cropped, rigid, params0, static_grid2world=g2w,
                         moving_grid2world=g2w, starting_affine=starting_affine)

    moving_[moving_ == moving_.min()] = fill_value
    tx.domain_grid2world[:2, -1] = -coms
    warped = tx.transform(moving_, sampling_grid_shape=moving_.shape)
    return warped, tx


def get_cropped_fish(image, phi, dydx, crop_window):
    # given an image, a rotation angle, and a shift, apply the shift, then the rotation, then crop around the
    # center of the transformed image and apply a binary mask, returning an image of the fish nervous system on a
    # a background of 0s

    from skimage.transform import rotate
    from numpy import roll, round, array, rad2deg
    from skimage.morphology import remove_small_objects

    shifted = roll(image, round(dydx).astype('int'), axis=(0, 1))
    rotated = rotate(shifted, angle=rad2deg(phi), mode='wrap', preserve_range=True, order=3)

    mid = array(rotated.shape) // 2
    window_y, window_x = crop_window

    # we take the transpose to cancel the effect of this kind of indexing
    crop = rotated[mid[0] + window_y, mid[1] + window_x].T
    # since we're cropped, we don't need to worry too much about about background removal
    # this may let some crap through, but that's ok
    mask = remove_small_objects(crop > long_tail_threshold(crop, shrink_factor=.3), min_size=100)

    return (crop * mask).astype('int16')


# point the tail of the fish toward the origin, parallel to the x-axis
# im must be a binarized fish mask
def orient_tail(im):
    from numpy import pi, array, abs, argmin
    from scipy.ndimage.measurements import center_of_mass
    from scipy.sparse import issparse

    im_ = im.copy()

    if issparse(im_):
        im_ = array(im_.todense())

    phi = 0
    dydx = (0, 0)

    # if we can't find a fish or a brain, return all 0s
    if not im_.any():
        return phi, dydx

    brain = get_brain(im_, return_sparse=False)

    # if we can't find a fish or a brain, return all 0s
    if not brain.any():
        return phi, dydx

    raw_com = array(center_of_mass(im_))
    brain_com = array(center_of_mass(brain))
    brain_center = remotest_point(im_)
    y_, x_ = brain_center

    tail_y, tail_x = brain_com - raw_com

    brain_tail_angle = cart2pol(tail_x, tail_y)[-1]

    # Eigenvector of largest eigenvalue corresponds to orientation of the long axis of the brain
    evals, evecs = eigenvalues(brain)
    brain_angle = cart2pol(evecs[0, 0], evecs[1, 0])[-1]
    candidate_phis = array([brain_angle, brain_angle + pi])

    # Eigenvector of brain is ambiguous between tail to the left and tail to the right. Pick angle that
    # minimizes angular distance from brain-tail angle
    which_angle = argmin([abs(angle_difference(brain_tail_angle, candidate)) for candidate in candidate_phis])
    phi = candidate_phis[which_angle]

    # this vector translates the center of the brain to the center of the image
    dydx = (im_.shape[0] / 2) - y_, (im_.shape[1] / 2) - x_

    return phi, dydx

import numpy as np

from spatialize import SpatializeError


def flatten_grid_data(xi):
    try:
        if len(xi) == 1:
            ng_xi = np.column_stack((xi.flatten()))
        elif len(xi) == 2:
            (grid_x, grid_y) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten()))
        elif len(xi) == 3:
            (grid_x, grid_y, grid_z) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))
        elif len(xi) == 4:
            (grid_x, grid_y, grid_z, grid_t) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), grid_t.flatten()))
        else:
            raise Exception
    except:
        raise SpatializeError("No grid data positions found")
    return ng_xi, grid_x.shape


class BilateralFilteringFusion:
    def __init__(self, cube, final_bands=1, c=0.1, c1=1. / 16., c2=0.05, m=10):
        self.cube = np.ma.array(cube, fill_value=0, mask=np.isnan(cube)).filled().astype(np.float32)
        self.final_bands = final_bands
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.m = m

        self.weights = None

    def filter_cube(self):
        # OpenCV is imported here, the only place it is used, so that importing spatialize does not need it
        from cv2 import bilateralFilter
        local_cube = self.cube
        filtered_cube = np.empty(local_cube.shape, dtype=np.float32)
        m, n, k = local_cube.shape
        sigma_s = self.c1 * min(m, n)
        sigma_r = np.array(self.c2 * (local_cube.max(axis=(0, 1)) - local_cube.min(axis=(0, 1))))
        for i, img in enumerate(local_cube.swapaxes(0, 2).swapaxes(1, 2)):
            filtered_cube[:, :, i] = bilateralFilter(img, 5, sigma_r[i], sigma_s)
        return filtered_cube

    def eval(self):
        local_cube = self.cube
        m, n, l = local_cube.shape
        p = int(np.ceil(local_cube.shape[2] / self.final_bands))
        filtered_cube = self.filter_cube()

        weights = np.empty_like(local_cube)
        for i in range(self.final_bands):
            local_cube_range = local_cube[:, :, i * p:(i + 1) * p]
            filtered_cube_range = filtered_cube[:, :, i * p:(i + 1) * p]
            abs_difference = np.abs(local_cube_range - filtered_cube_range) + self.c
            abs_difference_sum = abs_difference.sum(axis=2)
            weights[:, :, i * p:(i + 1) * p] = abs_difference / abs_difference_sum[:, :, np.newaxis]

        self.weights = weights
        lin_comb = np.zeros(shape=(m, n, self.final_bands))
        for i in range(self.final_bands):
            lin_comb[:, :, i] = (self.weights[:, :, i * p:(i + 1) * p] * local_cube[:, :, i * p:(i + 1) * p]).sum(
                axis=2)
        lin_comb[lin_comb > 65535.0] = 65535.0
        lin_comb[lin_comb < 0.0] = 0.0
        return lin_comb

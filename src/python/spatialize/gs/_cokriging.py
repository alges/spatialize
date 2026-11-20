import numpy as np, pandas as pd, json
from scipy.spatial import cKDTree

class Cokriging(object):

    def __init__(self, init_params):
        self.params = {
            'current_target_idx': -1
        }
        # validation
        self.set_init_params_and_validate(init_params)

        # filter the data and top_cuts
        self.filter_data()

        # transformation matrix for searching
        self.search_transformation = self.transformation_matrix(
            self.params['nb_radius'],
            self.params['nb_angles']
        )

        # calculate transformation for variograms
        self.variogram_transformation = []
        for i in range(len(self.params['variogram']['model'])):
            # use just the ranges and angles
            self.variogram_transformation.append(self.transformation_matrix(
                self.params['variogram']['model'][i]['ranges'],
                self.params['variogram']['model'][i]['angles']
            ))

        # create KDTree
        self.kdtree = cKDTree(np.dot(self.samples_coords, self.search_transformation))

    @staticmethod
    def from_json(path):
        with open(path, 'r') as f:
            params = json.loads(f.read())            

    @staticmethod
    def distance(p):
        # ones matrix with same size
        o = np.ones(p.shape)
        # every slice is a component, 0->x 1->y 2->z
        # column 1 [x|y|z] component, column 2 -1 value
        m1 = np.array([p, -1.0*o]).transpose((2,1,0))
        # row 1 [x|y|z] component, row 2 -1 value
        m2 = np.array([o, p]).transpose((2,0,1))
        # dot product in every slice (by component) is a matrix m_ijk of difference between p_i and p_j in component k, then power the matrix by 2
        quad_diffs = np.array([np.dot(m1[0,:,:], m2[0,:,:]), np.dot(m1[1,:,:], m2[1,:,:]), np.dot(m1[2,:,:], m2[2,:,:])])**2
        # sum the component slices and get the root
        d = np.sqrt(np.sum(quad_diffs, axis=0))

        return d

    def set_init_params_and_validate(self, i_params):
        # ask for sample_coords
        if 'samples_coords' not in i_params:
            raise Exception('Parameter \'samples_coords\' needed (Nx3 array, N: number of samples)')
        self.samples_coords = np.array(i_params['samples_coords'], dtype=np.float64)
        # ask for sample_values
        if 'samples_values' not in i_params:
            raise Exception('Parameter \'samples_values\' needed (NxM array, N: number of samples, M: number of variables)')
        self.samples_values = np.array(i_params['samples_values'], dtype=np.float64)
        # store sample_dhid if it was given
        self.samples_dhid = i_params['samples_dhid'] if 'samples_dhid' in i_params else None
        # coords and values must have same length
        if self.samples_coords.shape[0] != self.samples_values.shape[0]:
            raise Exception('Parameters \'samples_coords\'(%d) and \'samples_values\'(%d) do not have same length'%(self.samples_coords.shape[0], self.samples_values.shape[0]))
        if (self.samples_dhid is not None) and self.samples_coords.shape[0] != self.samples_dhid.shape[0]:
            raise Exception('Parameters \'samples_coords\'(%d) and \'samples_dhid\'(%d) do not have same length'%(self.samples_coords.shape[0], self.samples_dhid.shape[0]))

        # ask for kriging type
        if 'kriging_type' not in i_params:
            raise Exception('Parameter \'kriging_type\' needed (\'simple\' or \'ordinary\')')
        self.params['kriging_type'] = i_params['kriging_type'].lower()
        # if it is simple, it needs means
        if self.params['kriging_type'] == 'simple':
            # ask for means
            if 'kriging_mean' not in i_params:
                raise Exception('Parameter \'kriging_mean\' needed (N array, N: number of variables)')
            self.params['kriging_mean'] = np.array(i_params['kriging_mean'], dtype=np.float64)
            # means and values must have same dimension
            if self.params['kriging_mean'].shape[0] != self.samples_values.shape[1]:
                raise Exception('Parameters \'kriging_mean\'(%d) and \'samples_values\'(%d) do not have same dimension'%(self.params['kriging_mean'].shape[0], self.samples_values.shape[1]))
        elif self.params['kriging_type'] == 'ordinary':
            pass # it is ok
        else:
            raise Exception('Parameter \'kriging_type\' = \'%s\' is not allowed' % self.params['kriging_type'])

        # ask for limits, if not set min and max for float64
        if 'limits' not in i_params:
            i_params['limits'] = [np.finfo(np.float64).min, np.finfo(np.float64).max]
        self.params['limits'] = np.array(i_params['limits'], dtype=np.float64)
        # grid parameter and samples must have same dimension
        if self.params['limits'].shape[0] != 2:
            raise Exception('Parameter \'limits\'(%d) does not have 2 values (min and max)'%self.params['limits'].shape[0])

        # ask for top_cut, if not set max for float64
        if 'top_cut' not in i_params:
            i_params['top_cut'] = [np.finfo(np.float64).max]*self.samples_values.shape[1]
        self.params['top_cut'] = np.array(i_params['top_cut'], dtype=np.float64)
        # grid parameter and samples must have same dimension
        if self.params['top_cut'].shape[0] != self.samples_values.shape[1]:
            raise Exception('Parameters \'top_cut\'(%d) and \'samples_values\'(%d) do not have same dimension'%(self.params['top_cut'].shape[0], self.samples_values.shape[1]))

        # ask for neighborhood radius
        if 'nb_radius' not in i_params:
            raise Exception('Parameter \'nb_radius\' needed (3 array)')
        self.params['nb_radius'] = np.array(i_params['nb_radius'], dtype=np.float64)
        # neighborhood and samples must have same dimension
        if self.params['nb_radius'].shape[0] != 3:
            raise Exception('Parameters \'nb_radius\'(%d) does not have 3 dimensions'%self.params['nb_radius'].shape[0])

        # ask for neighborhood angles
        if 'nb_angles' not in i_params:
            raise Exception('Parameter \'nb_angles\' needed (3 array)')
        self.params['nb_angles'] = np.array(i_params['nb_angles'], dtype=np.float64)
        # neighborhood and samples must have same dimension
        if self.params['nb_angles'].shape[0] != 3:
            raise Exception('Parameters \'nb_angles\'(%d) does not have 3 dimensions'%self.params['nb_angles'].shape[0])

        # ask for neighborhood angles
        if 'nb_ndata' not in i_params:
            raise Exception('Parameter \'nb_ndata\' needed (2 array)')
        self.params['nb_ndata'] = np.array(i_params['nb_ndata'], dtype=np.int16)
        # neighborhood and samples must have same dimension
        if self.params['nb_ndata'].shape[0] != 2:
            raise Exception('Parameters \'nb_ndata\'(%d) does not have 2 values (min and max)'%self.params['nb_ndata'].shape[0])

        # ask for use octants in neighborhood
        self.params['nb_octant'] = False if 'nb_octant' not in i_params else i_params['nb_octant']

        # ask for missing_value
        self.params['missing_value'] = np.nan if 'missing_value' not in i_params else i_params['missing_value']

        # set variogram (TODO: validate LCM)
        self.params['variogram'] = i_params['variogram']

    def set_execute_params_and_validate(self, e_params):
        # ask for discretization, if not it is punctual
        if 'discretization' not in e_params:
            e_params['discretization'] = [1, 1, 1]
        self.params['discretization'] = np.array(e_params['discretization'], dtype=np.int16)
        # grid parameter and samples must have same dimension
        if self.params['discretization'].shape[0] != 3:
            raise Exception('Parameter \'discretization\'(%d) does not have 3 dimensions'%self.params['discretization'].shape[0])

        # ask for block_size
        if 'block_size' not in e_params:
            # when discretization is punctual ([1,1,1]) then does not matter the block size
            if (self.params['discretization'] == np.array([1, 1, 1], dtype=np.int16)).all():
                e_params['block_size'] = [1, 1, 1]
            else:
                raise Exception('Parameter \'block_size\' needed (3 array)')
        self.params['block_size'] = np.array(e_params['block_size'], dtype=np.float64)
        # grid parameter and samples must have same dimension
        if self.params['block_size'].shape[0] != 3:
            raise Exception('Parameter \'block_size\'(%d) does not have 3 dimensions'%self.params['block_size'].shape[0])

        # ask for target
        if 'target' not in e_params:
            raise Exception('Parameter \'target\' needed (\'scattered\', \'grid\', \'leave_one_out_sample\' or \'leave_one_out_drillhole\')')
        self.params['target'] = e_params['target'].lower()
        # if it is scattered, it needs query_coords
        if self.params['target'] == 'scattered':
            # ask for means
            if 'target_coords' not in e_params:
                raise Exception('Parameter \'target_coords\' needed (Nx3 array, N: number of target points')
            self.target_coords = np.array(e_params['target_coords'], dtype=np.float64)
            # querys and samples must have same dimension
            if self.target_coords.shape[1] != 3:
                raise Exception('Parameter \'target_coords\'(%d) does not have 3 dimensions'%self.target_coords.shape[1])
        # if it is grid, it needs grid_origin, grid_nodes (grid_size will be block size)
        elif self.params['target'] == 'grid':
            # ask for grid_origin
            if 'grid_origin' not in e_params:
                raise Exception('Parameter \'grid_origin\' needed (3 array)')
            self.params['grid_origin'] = np.array(e_params['grid_origin'], dtype=np.float64)
            # grid parameter and samples must have same dimension
            if self.params['grid_origin'].shape[0] != 3:
                raise Exception('Parameter \'grid_origin\'(%d) does not have 3 dimensions'%self.params['grid_origin'].shape[0])
            # ask for grid_nodes
            if 'grid_nodes' not in e_params:
                raise Exception('Parameter \'grid_nodes\' needed (3 array)')
            self.params['grid_nodes'] = np.array(e_params['grid_nodes'], dtype=np.int16)
            # grid parameter and samples must have same dimension
            if self.params['grid_nodes'].shape[0] != 3:
                raise Exception('Parameter \'grid_nodes\'(%d) does not have 3 dimensions'%self.params['grid_nodes'].shape[0])
        elif self.params['target'] == 'leave_one_out_sample':
            pass # it is ok
        elif self.params['target'] == 'leave_one_out_drillhole':
            if self.samples_dhid is None:
                raise Exception('Parameter \'samples_dhid\' must be given for \'target\' = \'leave_one_out_drillhole\'')
            self.sample_id_by_dh_id = {}
            for i,dhid in enumerate(self.samples_dhid):
                if dhid not in self.sample_id_by_dh_id:
                    self.sample_id_by_dh_id[dhid] = []
                self.sample_id_by_dh_id[dhid].append(i)
        else:
            raise Exception('Parameter \'targer\' = \'%s\' is not allowed' % self.params['target'])

    def initialize(self):
        if self.params['target'] == 'grid':
            # generate target_coords
            self.create_grid()
        elif self.params['target'] in ['leave_one_out_sample', 'leave_one_out_drillhole']:
            # set samples cords as target
            self.target_coords = self.samples_coords[...]

        # block discretization
        self.discretize_block()

    def create_grid(self):
        # array to store points
        coords = []
        for k in range(self.params['grid_nodes'][2]):
            for j in range(self.params['grid_nodes'][1]):
                for i in range(self.params['grid_nodes'][0]):
                    # get coordinates for point to estimate
                    coords.append([
                        self.params['grid_origin'][0] + (i+.5)*self.params['block_size'][0],
                        self.params['grid_origin'][1] + (j+.5)*self.params['block_size'][1],
                        self.params['grid_origin'][2] + (k+.5)*self.params['block_size'][2],
                    ])
        self.target_coords = np.array(coords, dtype=np.float64)

    def discretize_block(self):
        # limits
        _x = (self.params['block_size'][0] / 2.0)*(1.0 + 1.0 / self.params['discretization'][0])
        _y = (self.params['block_size'][1] / 2.0)*(1.0 + 1.0 / self.params['discretization'][1])
        _z = (self.params['block_size'][2] / 2.0)*(1.0 + 1.0 / self.params['discretization'][2])
        # separations
        dx = list(np.linspace(-_x, _x, self.params['discretization'][0]+2))[1:-1]
        dy = list(np.linspace(-_y, _y, self.params['discretization'][1]+2))[1:-1]
        dz = list(np.linspace(-_z, _z, self.params['discretization'][2]+2))[1:-1]
        # sub arrays sorted in order to make a grid
        zs = np.array([[_z]*(self.params['discretization'][0]*self.params['discretization'][1]) for _z in dz]).flatten()
        ys = np.array([[_y]*self.params['discretization'][0] for _y in dy]*self.params['discretization'][2]).flatten()
        xs = np.array(list(dx)*(self.params['discretization'][1]*self.params['discretization'][2])).flatten()
        # create the list of points
        self.block_grid = np.array([xs, ys, zs], dtype=np.float64).transpose()
        self.block_grid.reshape(3,np.prod(self.params['discretization']))

    def filter_data(self):
        # filter the missing
        idx = np.where(self.samples_values == self.params['missing_value'])
        if len(idx[0]) > 0:
            self.samples_values[idx] = np.nan

        # filter all values using given limits
        idx = np.where(self.samples_values < self.params['limits'][0])
        if len(idx[0]) > 0:
            self.samples_values[idx] = np.nan
        idx = np.where(self.params['limits'][1] < self.samples_values)
        if len(idx[0]) > 0:
            self.samples_values[idx] = np.nan

        # apply top cuts
        for i in range(len(self.samples_values)):
            # if there is a top_cut for variable
            if i < len(self.params['top_cut']):
                # set top_cut value for vaues over it
                idx = np.where(self.samples_values[:,i] > self.params['top_cut'][i])[0]
                self.samples_values[idx,i] = self.params['top_cut'][i]

    def transformation_matrix(self, ranges, angles):
        # set reduction matrix
        reduction = np.diag(1.0/np.array(ranges))
        # angles to rad
        deg2rad = np.pi/180.0
        a = (90.0-angles[0])*deg2rad
        b = -angles[1]*deg2rad
        c = angles[2]*deg2rad
        # sin and cos
        ca = np.cos(a)
        sa = np.sin(a)
        cb = np.cos(b)
        sb = np.sin(b)
        cc = np.cos(c)
        sc = np.sin(c)
        # rotation matrix
        rotation = np.empty((3,3), dtype=np.float64)
        rotation[0,0] = cb * ca
        rotation[0,1] = cb * sa
        rotation[0,2] = -sb
        rotation[1,0] = -cc * sa + sc * sb * ca
        rotation[1,1] = cc * ca + sc * sb * sa
        rotation[1,2] = sc * cb
        rotation[2,0] = sc * sa + cc * sb * ca
        rotation[2,1] = -sc * ca + cc * sb * sa
        rotation[2,2] = cc * cb
        # get whole transformation
        return np.dot(reduction, rotation).transpose()

    def get_neighbors(self, p):
        # is missing value predicate
        def is_missing(index, var):
            return np.isnan(self.samples_values[index, var])

        # transform query point
        q = np.dot(p, self.search_transformation)
        # get data in the search radii
        nb = np.array(self.kdtree.query_ball_point(q, 1))

        # if leave one out, must drop the query point from the neighbors
        if self.params['target'] == 'leave_one_out_sample':
            nb = np.array([e for e in nb if e != self.params['current_target_idx']])
        elif self.params['target'] == 'leave_one_out_drillhole':
            curr_dhid = self.samples_dhid[self.params['current_target_idx']]
            curr_dh = self.sample_id_by_dh_id[curr_dhid]
            nb = np.array([e for e in nb if e not in curr_dh])

        # if no neighbors just return empty
        if len(nb) == 0:
            return [[]]*self.samples_values.shape[1]

        dif  = np.dot(self.samples_coords[nb], self.search_transformation) - q
        # get neighbors position, transform them and get difference
        dist = np.sum(dif**2, axis=1)
        # sort
        order = np.argsort(dist)
        # apply to the nb and dif
        nb = nb[order]
        dif = dif[order]
        # number of variables
        n_var = self.samples_values.shape[1]

        # if octant is selected filter by them
        if self.params['nb_octant']:
            # count and store the neighbors
            per_octant = np.zeros((n_var, 8))
            selected_points = [[] for i in range(n_var)]
            # visit every neighbor
            for i, idx in enumerate(nb):
                # get octant
                oct = (0 if dif[i,0]<0 else 1) + (0 if dif[i,1]<0 else 1)*2 + (0 if dif[i,2]<0 else 1)*4
                # ask if is full, and no missing, for every variable
                for j in range(n_var):
                    if not is_missing(idx, j) and per_octant[j, oct] < self.params['nb_ndata'][1]:
                        # add it to the list
                        per_octant[j, oct] += 1
                        selected_points[j].append(idx)
            return selected_points
        else:
            # get the not missing values for every variable and crop if needed
            return [nb[np.where(~np.isnan(self.samples_values[nb, j]))[0][:self.params['nb_ndata'][1]]] for j in range(n_var)]

    def estimation(self, xs, nbs):
        # covariance using the given structure
        def structure(st, h):
            if st == 'spherical':
                hp = np.min(np.array([h,np.ones(h.shape)]), axis=0)
                return 1 - 1.5*hp + 0.5*(hp**3)
            elif st == 'exponential':
                return np.exp(-3*h)
            elif st == 'cubic':
                hp = np.min(np.array([h,np.ones(h.shape)]), axis=0)
                return 1 - 7*(hp**2) + (35/4)*(hp**3) - (7/2)*(hp**5) + (3/4)*(hp**7)
            elif st == 'gaussian':
                return np.exp(-3*h**2)
            else:
                return 1

        # number of variables, samples and discretization points
        n_vars = len(nbs)
        n_discretization = len(xs)

        # neighbors samples coords
        nb_coords = self.samples_coords[np.concatenate(tuple(nbs))]
        nb_vars = np.concatenate(tuple([i]*len(nbs[i]) for i in range(n_vars)))

        # new xs
        _xs = np.concatenate(tuple([xs]*n_vars), axis=0)
        _vs = np.array([[i]*len(xs) for i in range(n_vars)]).flatten()

        # all points
        points = np.concatenate((nb_coords, _xs), axis=0)
        vars = np.int64(np.concatenate((nb_vars, _vs), axis=0))

        # starts with nugget
        c = np.array([[self.params['variogram']['nugget'][vars[i]][vars[j]] if (points[i]==points[j]).all() else 0 for j in range(len(vars))] for i in range(len(vars))])
        # c = np.array([[self.params['variogram']['nugget'][vars[i]][vars[j]] for j in range(len(vars))] for i in range(len(vars))])

        # sum every nested structure aportation
        for _i in range(len(self.params['variogram']['model'])):
            # get points rotated and scaled by variogram ranges and angles
            rps = np.dot(points, self.variogram_transformation[_i])
            # distances between points
            dists = Cokriging.distance(rps)
            # get covariance
            cov = structure(self.params['variogram']['model'][_i]['type'], dists)
            sill = np.array([[self.params['variogram']['model'][_i]['sill'][vars[i]][vars[j]] for j in range(len(vars))] for i in range(len(vars))])
            c = c + cov*sill
        # get the k and k0 values ti solve de kriging system
        k = c[:len(nb_vars),:len(nb_vars)]
        k0 = c[:len(nb_vars),len(nb_vars):]
        k1 = c[len(nb_vars):,len(nb_vars):]

        # if it is block kriging (discretization has mode than 1 point)
        if n_discretization>1:
            # apply average over discretization points
            k0 = np.array([np.mean(k0[:,i*n_discretization:(i+1)*n_discretization], axis=1) for i in range(n_vars)]).transpose()
            k1 = np.array([[np.mean(k1[j*n_discretization:(j+1)*n_discretization,i*n_discretization:(i+1)*n_discretization]) for i in range(n_vars)] for j in range(n_vars)])

        if self.params['kriging_type'] == 'simple':
            # solve the system k*weights = k0
            k_inv = np.linalg.inv(k)
            weights = np.dot(k_inv, k0)
            sigma = k1 - np.dot(weights.transpose(), k0) - np.dot(k0.transpose(), weights) + np.dot(np.dot(weights.transpose(), k), weights)
        elif self.params['kriging_type'] == 'ordinary':
            # lines to add for perform an ordinary kriging
            lat = np.zeros((len(nb_vars), n_vars))
            # set 1 to corresponding values
            cnt = 0
            for i in range(n_vars):
                lat[cnt:cnt+len(nbs[i]),i] = 1
                cnt += len(nbs[i])

            # attach the new values to k matrix
            k = np.concatenate((k, lat), axis=1)
            # extend the lat matrix with a n_vars x nvars zero matrix
            lat = np.concatenate((lat, np.zeros((n_vars, n_vars))), axis=0)
            # attach the extended lat matrix
            k = np.concatenate((k, lat.transpose()), axis=0)
            # attach new values for the k0 matrix
            k0 = np.concatenate((k0, np.eye(n_vars)), axis=0)
            # solve the extended system k*weights = k0
            weights = np.dot(np.linalg.inv(k), k0)[:-n_vars,:]
            sigma = k1 - np.dot(weights.transpose(), k0[:-n_vars]) - np.dot(k0[:-n_vars].transpose(), weights) + np.dot(np.dot(weights.transpose(), k[:-n_vars,:-n_vars]), weights)

        return weights, sigma

    def __call__(self, execute_params):
        # validate execution parameters
        self.set_execute_params_and_validate(execute_params)
        # create some stuff
        self.initialize()

        # prediction and variance arrays
        self.prediction = np.ones((self.target_coords.shape[0], self.samples_values.shape[1]))*self.params['missing_value']
        self.variance = np.ones((self.target_coords.shape[0], self.samples_values.shape[1]))*self.params['missing_value']
        self.nbrs = np.zeros((self.target_coords.shape[0], self.samples_values.shape[1]))

        # visit every position
        total = self.target_coords.shape[0]
        for idx, ncoord in enumerate(self.target_coords):
            # for leave_one_out target
            self.params['current_target_idx'] = idx
            if idx%1000==999: print('%d/%d'%(idx+1, total))

            # look up for neighbors
            neighbors = self.get_neighbors(ncoord)
            self.nbrs[idx,:] = [len(n) for n in neighbors]
            # if no neighbors step over it
            if np.sum(self.nbrs[idx,:]) < self.params['nb_ndata'][0]:
                continue
            # get the block discratization
            ncoords = ncoord + self.block_grid
            # estimate
            try:
                weights, sigma = self.estimation(ncoords, neighbors)
                # retrieve the values and concatenate first var1, then var2 and so on
                values = np.concatenate(tuple(self.samples_values[nb, i] for i,nb in enumerate(neighbors)))
                # prediction and variance
                self.prediction[idx,:] = np.dot(values, weights)
                self.variance[idx,:] = np.diag(sigma)
            except Exception as e:
                self.nbrs[idx,:] = -1

        return(self.prediction, self.variance, self.nbrs)

    
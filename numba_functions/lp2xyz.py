


def lph2xyz(l, p, h, ):
    # LPH2XYZ   Convert radar line/pixel coordinates into Cartesian coordinates.
    #   XYZ=LPH2XYZ(LINE,PIXEL,HEIGHT,IMAGE,ORBFIT) converts the radar coordinates
    #   LINE and PIXEL into Cartesian coordinates XYZ. HEIGHT contains the height
    #   of the pixel above the ellipsoid. IMAGE contains the image metadata
    #   and ORBFIT the orbit fit, proceduced respectively by the METADATA and
    #   ORBITFIT processing_steps.
    #
    #   [XYZ,SATVEC]=LPH2XYZ(...) also outputs the corresponding satellite
    #   position and velocity, with SATVEC a matrix with in it's columns the
    #   position X(m), Y(m), Z(m) and velocity X_V(m/s), Y_V(m/s), Z_V(m/s),
    #   with rows corresponding to rows of XYZ.
    #
    #   [...]=LPH2XYZ(...,'VERBOSE',0,'MAXITER',10,'CRITERPOS',1e-6,'ELLIPSOID',
    #   [6378137.0 6356752.3141]) includes optional input arguments VERBOSE,
    #   MAXITER, CRITERPOS and ELLIPSOID  to overide the default verbosity level,
    #   interpolator exit criteria and ellipsoid. The defaults are the WGS-84
    #   ELLIPSOID=[6378137.0 6356752.3141], MAXITER=10, CRITERPOS=1e-6 and
    #   VERBOSE=0.
    #
    #   Example:
    #       [image, orbit] = metadata('master.res');
    #       orbfit = orbitfit(orbit);
    #       xyz = lph2xyz(line,pixel,0,image,orbfit);
    #
    #   See also METADATA, ORBITFIT, ORBITVAL, XYZ2LP and XYZ2T.
    #
    #   (c) Petar Marinkovic, Hans van der Marel, Delft University of Technology, 2007-2014.

    #   Created:    20 June 2007 by Petar Marinkovic
    #   Modified:   13 March 2014 by Hans van der Marel
    #                - added description and input argument checking
    #                - use orbit fitting procedure
    #                - added output of satellite position and velocity
    #                - original renamed to LPH2XYZ_PM
    #                6 April 2014 by Hans van der Marel
    #                - improved handling of optional parameters
    #                5 July 2017 by Gert Mulder
    #                - converted to python code
    #                - created read of .res files
    #                - vectorized to optimize for speed

    if len(self.lines) == 0 or len(self.pixels) == 0:
        print('First define for which pixels or which azimuth/range times you want to compute the xyz coordinates')
        return
    if len(self.height) == 0:
        print('First find the heights of the invidual pixels. This can be done using the create DEM function')
    if len(self.height) == 0:
        print('There is no height data loaded!')
        return

    ell_a = self.ellipsoid[0]
    ell_b = self.ellipsoid[1]
    ell_e2 = 1 - ell_b ** 2 / ell_a ** 2

    # Some preparations to get the start conditions
    h = np.mean(self.height)

    height = np.ravel(self.height)
    ell_a_2 = (ell_a + height) ** 2  # Preparation for distance on ellips with certain height
    ell_b_2 = (ell_b + height) ** 2  # Preparation for distance on ellips with certain height
    Ncenter = ell_a / np.sqrt(1 - ell_e2 * (np.sin(self.center_phi) ** 2))
    scenecenterx = (Ncenter + h) * np.cos(self.center_phi) * np.cos(self.center_lambda)
    scenecentery = (Ncenter + h) * np.cos(self.center_phi) * np.sin(self.center_lambda)
    scenecenterz = (Ncenter + h - ell_e2 * Ncenter) * np.sin(self.center_phi)

    # These arrays are only in the azimuth direction
    possatx = self.xyz_orbit[0, :]
    possaty = self.xyz_orbit[1, :]
    possatz = self.xyz_orbit[2, :]
    velsatx = self.vel_orbit[0, :]
    velsaty = self.vel_orbit[1, :]
    velsatz = self.vel_orbit[2, :]

    # First guess
    num = len(self.lines) * len(self.pixels)
    posonellx = np.ones(num) * scenecenterx
    posonelly = np.ones(num) * scenecentery
    posonellz = np.ones(num) * scenecenterz

    # 1D id, 2D row and column ids (used to extract information
    az = np.arange(len(self.lines)).astype(np.int32)[:, None]
    az_id = np.ravel(az * np.ones((1, len(self.pixels)))).astype(np.int32)

    # Next parameter defines which points still needs another iteration to solve. If the precisions are met,
    # this point will be removed from the dataset.
    solve_ids = np.arange(len(self.lines) * len(self.pixels)).astype(np.int32)

    # distance to pixel
    range_dist = np.ravel((self.sol * self.ra_times[None, :] / 2) ** 2 * np.ones((len(self.lines), 1)))

    for iterate in range(self.maxiter):

        # Distance of orbit points with start point
        dsat_Px = np.take(posonellx, solve_ids) - np.take(possatx, az_id)
        dsat_Py = np.take(posonelly, solve_ids) - np.take(possaty, az_id)
        dsat_Pz = np.take(posonellz, solve_ids) - np.take(possatz, az_id)

        # Equations 1. range line perpendicular to orbit
        #           2. range time times speed of light same as distance orbit to point
        #           3. point on ellipsoid
        equations = np.zeros(shape=(3, len(solve_ids)))

        equations[0, :] = -(np.take(velsatx, az_id) * dsat_Px + np.take(velsaty, az_id) *
                            dsat_Py + np.take(velsatz, az_id) * dsat_Pz)
        equations[1, :] = -(dsat_Px ** 2 + dsat_Pz ** 2 + dsat_Py ** 2 - range_dist)  # Add average atmospheric delay?
        equations[2, :] = -((np.take(posonellx, solve_ids) ** 2 + np.take(posonelly, solve_ids) ** 2) /
                            ell_a_2 + (np.take(posonellz, solve_ids) ** 2 / ell_b_2) - 1)

        # derivatives of 3 components for linearization
        derivatives = np.zeros(shape=(3, 3, len(solve_ids)))

        derivatives[0, 0, :] = np.take(velsatx, az_id)
        derivatives[1, 0, :] = np.take(velsaty, az_id)
        derivatives[2, 0, :] = np.take(velsatz, az_id)
        derivatives[0, 1, :] = 2 * dsat_Px
        derivatives[1, 1, :] = 2 * dsat_Py
        derivatives[2, 1, :] = 2 * dsat_Pz
        derivatives[0, 2, :] = (2 * np.take(posonellx, solve_ids)) / ell_a_2
        derivatives[1, 2, :] = (2 * np.take(posonelly, solve_ids)) / ell_a_2
        derivatives[2, 2, :] = (2 * np.take(posonellz, solve_ids)) / ell_b_2
        del dsat_Px, dsat_Py, dsat_Pz

        # Solve system of equations
        solpos = np.linalg.solve(derivatives.swapaxes(0, 2), equations.swapaxes(0, 1)).swapaxes(0, 1)

        # Update solution
        posonellx[solve_ids] += solpos[0, :]
        posonelly[solve_ids] += solpos[1, :]
        posonellz[solve_ids] += solpos[2, :]
        del derivatives

        # Check which ids are close enough
        not_finished = np.ravel(np.argwhere(((np.abs(solpos[0, :]) < self.criterpos) *
                                             (np.abs(solpos[1, :]) < self.criterpos) *
                                             (np.abs(solpos[2, :]) < self.criterpos)) == False))

        # If all points are found we can stop the iteration
        if len(not_finished) == 0:
            # print('All points located within ' + str(iterate + 1) + ' iterations.')
            break

        # prepare for next iteration by removing values from these variables
        solve_ids = np.take(solve_ids, not_finished)
        az_id = np.take(az_id, not_finished)
        range_dist = np.take(range_dist, not_finished)
        ell_a_2 = np.take(ell_a_2, not_finished)
        ell_b_2 = np.take(ell_b_2, not_finished)

        # If some point are not found within the iteration time, give a warning
        if iterate == self.maxiter - 1:
            print(str(len(solve_ids)) + 'did not converge within ' + str(
                self.maxiter) + ' iterations. Maybe use more iterations or less stringent criteria?')

    shp = (len(self.lines), len(self.pixels))
    self.x = np.reshape(posonellx, shp)
    del posonellx
    self.y = np.reshape(posonelly, shp)
    del posonelly
    self.z = np.reshape(posonellz, shp)
    del posonellz

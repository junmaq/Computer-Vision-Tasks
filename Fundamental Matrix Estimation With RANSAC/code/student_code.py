import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # A = np.zeros((len(points_3d)*2,11))
    A = np.zeros((len(points_3d)*2,11))
    b = np.zeros((len(points_3d)*2))
    
    for i in range(len(points_3d)):
        
        A[2*i,0]=points_3d[i,0];A[2*i+1,4]=points_3d[i,0]
        A[2*i,1]=points_3d[i,1];A[2*i+1,5]=points_3d[i,1]
        A[2*i,2]=points_3d[i,2];A[2*i+1,6]=points_3d[i,2]
        A[2*i,3]=1;A[2*i+1,7]=1
        
        
        A[2*i,8]=-points_3d[i][0]*points_2d[i,0]
        A[2*i,9]=-points_3d[i][1]*points_2d[i,0]
        A[2*i,10]=-points_3d[i][2]*points_2d[i,0]
        
        A[2*i+1,8]=-points_3d[i,0]*points_2d[i,1]
        A[2*i+1,9]=-points_3d[i,1]*points_2d[i,1]
        A[2*i+1,10]=-points_3d[i,2]*points_2d[i,1]
        b[2*i]=points_2d[i][0]
        b[2*i+1]=points_2d[i][1]
    
    P=np.linalg.pinv(A).dot(b)
    M[0,:]=P[:4]
    M[1,:]=P[4:8]
    M[2,:3]=P[8:]
    M[2,3]=1
    ###########################################################################

    # raise NotImplementedError('`calculate_projection_matrix` function in ' +
        # '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    # raise NotImplementedError('`calculate_camera_center` function in ' +
        # '`student_code.py` needs to be implemented')
    cc=-np.linalg.inv(M[:,:3]).dot(M[:,3])
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b, mode=None):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

    # raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
        # '`student_code.py` needs to be implemented')

    ###########################################################################
    # calcualting and subtracting centres
    ca=np.mean(points_a,axis=0)
    cb=np.mean(points_b,axis=0)
    
    points_a_mean=points_a-ca
    points_b_mean=points_b-cb
    
    std_a=np.std(points_a_mean,axis=0)
    std_b=np.std(points_b_mean,axis=0)
    
    points_a_std=points_a_mean/std_a
    points_b_std=points_b_mean/std_b
    
    norm_a=np.sqrt(np.mean(points_a_mean[:,0]**2+points_a_mean[:,1]**2))
    norm_b=np.sqrt(np.mean(points_b_mean[:,0]**2+points_b_mean[:,1]**2))
    
    points_a_norm=np.sqrt(2)*points_a_mean/norm_a
    points_b_norm=np.sqrt(2)*points_b_mean/norm_b    
    
    Ca=np.array([[1,0,-ca[0]],
                [0,1,-ca[1]],
                [0,0,1]])
    
    Cb=np.array([[1,0,-cb[0]],
                [0,1,-cb[1]],
                [0,0,1]])
    
    S_norma=np.array([[1/norm_a,0,0],
                     [0,1/norm_a,0],
                     [0,0,1]])
    
    S_normb=np.array([[1/norm_b,0,0],
                     [0,1/norm_b,0],
                     [0,0,1]])
    
    S_stda=np.array([[1/std_a[0],0,0],
                    [0,1/std_a[1],0],
                    [0,0,1]])
    
    S_stdb=np.array([[1/std_b[0],0,0],
                    [0,1/std_b[1],0],
                    [0,0,1]])
    
    
    A=np.zeros((len(points_a),8))
    
    if mode=="std":
        
        Ta=np.matmul(S_stda,Ca)
        Tb=np.matmul(S_stdb,Cb).transpose()
        points_ause=points_a_std
        points_buse=points_b_std
    
    elif mode=="norm":
        Ta=np.matmul(S_norma,Ca)
        Tb=np.matmul(S_normb,Cb).transpose()
        points_ause=points_a_norm
        points_buse=points_b_norm
    
    else:
        Ta=np.identity(3)
        Tb=np.identity(3)
        points_ause=points_a
        points_buse=points_b
    
    for i in range(len(points_a)):
        A[i,0]=points_ause[i,0]*points_buse[i,0];A[i,1]=points_ause[i,1]*points_buse[i,0]
        A[i,2]=points_buse[i,0];A[i,3]=points_ause[i,0]*points_buse[i,1]
        A[i,4]=points_ause[i,1]*points_buse[i,1];A[i,5]=points_buse[i,1]
        A[i,6]=points_ause[i,0];A[i,7]=points_ause[i,1]
    
    P=np.linalg.pinv(A).dot(-np.ones(len(points_a)))
    
    F[0,:]=P[:3]
    F[1,:]=P[3:6]
    F[2,:2]=P[6:8]
    F[2,2]=1
    U,S,V=np.linalg.svd(F)
    S[2]=0
    _=np.matmul(U,np.diag(S))
    F_=np.matmul(_,V)
    
    
    F_b=np.matmul(Tb,F_)
    F_ab=np.matmul(F_b,Ta)

    # F__=np.matmul(S_stda,F_)
    # _F=np.matmul(F__,S_stdb)
    ###########################################################################

    return F_ab

def ransac_fundamental_matrix(matches_a, matches_b,iters,th,n):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    # raise NotImplementedError('`ransac_fundamental_matrix` function in ' +
        # '`student_code.py` needs to be implemented')

    ###########################################################################
    
    mlen=0
    
    for i in range(iters):
         indices=np.random.choice(len(matches_a), size=n, replace=False)
         F=estimate_fundamental_matrix(matches_a[indices], matches_b[indices], mode="std")
         in_indices=[ np.abs(np.sum(np.array([b[0],b[1],1]).T.dot(F)*np.array([a[0],a[1],1]).T)) < th for a,b in zip(matches_a,matches_b) ]
         inliers_a=matches_a[in_indices]
         inliers_b=matches_b[in_indices]
         if len(inliers_a)>mlen:
             m_indices=in_indices
             mlen=len(inliers_a)
             print(mlen)
         
    best_F=estimate_fundamental_matrix(matches_a[m_indices],matches_b[m_indices],mode="std")
    inliers_a=matches_a[m_indices]
    inliers_b=matches_b[m_indices]
    return best_F,inliers_a,inliers_b
         
         
         
         
    ###########################################################################

    # return best_F, inliers_a, inliers_b
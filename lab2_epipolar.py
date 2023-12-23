import numpy as np
import cv2


def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    #computing the center of the points by computing the mean of x and y
    # TODO: find center of the set of points by computing mean of x & y
    # Evaluate the mean of x and y for all points
    center = np.mean(points,axis=0)
    
    # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    # Evaluate the distance from every point to the origin
    dist = np.linalg.norm(points - center, axis=1)

    # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    # Evaluate the scale factor
    s = np.sqrt(2) / np.mean(dist)
    
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# read image & put them in grayscale
# Read image from teh path: /home/emanuele/Desktop/AVG/TP_AVG/TP2_stereo/img
imgloc1 = "/home/emanuele/Desktop/AVG/TP_AVG/TP2_stereo/img/chapel00.png"
img1 = cv2.imread(imgloc1)  # queryImage
imgloc2 = "/home/emanuele/Desktop/AVG/TP_AVG/TP2_stereo/img/chapel01.png"
img2 = cv2.imread(imgloc2) # trainImage
# check if image is read successfully
assert img1 is not None, 'Failed to read image1'
assert img2 is not None, 'Failed to read image2'
# detect kpts & compute descriptor
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# match kpts
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# organize key points into matrix, each row is a point
query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

# normalize kpts
T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
print('T_query: ', T_query)
# TODO: apply T_query to query_kpts to normalize them
#converting manually to homogeneous coordinates
#normalized_query_kpts = np.concatenate((query_kpts, np.ones((normalized_query_kpts.shape[0], 1))), axis=1)
normalized_query_kpts = T_query @ homogenize(query_kpts).T 

T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts
# TODO: apply T_train to train_kpts to normalize them
#converting manually to homogeneous coordinates
#normalized_train_kpts = np.concatenate((train_kpts, np.ones((normalized_train_kpts.shape[0], 1))), axis=1)
normalized_train_kpts = T_train @ homogenize(train_kpts).T

print('normalized_query_kpts.shape: ', normalized_query_kpts.shape)
print('normalized_train_kpts.shape: ', normalized_train_kpts.shape)
#print('normalized_query_kpts: ', normalized_query_kpts)
#print('normalized_train_kpts: ', normalized_train_kpts)

# construct homogeneous linear equation to find fundamental matrix
# TODO: construct A according to Eq.(3) in lab subject
A = np.array([])  
# A should be a matrix with shape <num_pts, 9>
# each row of A is constructed from a pair of normalized kpts
# A = [x'x, x'y, x', y'x, y'y, y', x, y, 1]
x_prime = normalized_train_kpts[0,:]
print(x_prime.shape)
y_prime = normalized_train_kpts[1,:]
print(y_prime.shape)
x = normalized_query_kpts[0,:]
print(x.shape)
y = normalized_query_kpts[1,:]
print(y.shape)

A = np.array([x_prime*x, x_prime*y, x_prime, y_prime*x, y_prime*y, y_prime, x, y, np.ones(x.shape[0])]).T
print(A.shape)

# TODO: find vector f by solving A f = 0 using SVD
# hint: perform SVD of A using np.linalg.svd to get u, s, vh (vh is the transpose of v)
# hint: f is the last column of v
f = np.array([])  # TODO: find f

#performing SVD of A
u, s, vh = np.linalg.svd(A)
#f is the last column of v
f = vh[-1]


# arrange f into 3x3 matrix to get fundamental matrix F
F = f.reshape(3, 3)
print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

# TODO: force F to have rank 2
# hint: perform SVD of F using np.linalg.svd to get u, s, vh
# hint: set the smallest singular value of F to 0
# hint: reconstruct F from u, new_s, vh

#performing SVD of F
u, s, vh = np.linalg.svd(F)
#setting the smallest singular value of F to 0
#Print s to see the smallest singular value
print("s before: ", s)
s[-1] = 0
#Print s after setting the smallest singular value to 0
print("s after: ", s)
#reconstructing F from u, new_s, vh
F = u @ np.diag(s) @ vh

assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

# TODO: de-normlaize F
# hint: last line of Algorithme 1 in the lab subject
F = T_train.T @ F @ T_query

txtpath = "/home/emanuele/Desktop/AVG/TP_AVG/TP2_stereo/chapel.00.01.F"
F_gt = np.loadtxt(txtpath)
print(F - F_gt)


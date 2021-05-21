import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf


#---------------------------------------------------------------------
# Misc
#---------------------------------------------------------------------
def preprocess_img(x):
    x = 2 * x - 1
    x = tf.expand_dims(x, axis=0)
    return x


def norm(x, axis=None, keepdims=None, eps=1e-8):
    """
    Numerically stable norm.
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims) + eps)
    #return tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims)


#---------------------------------------------------------------------
# Brushstrokes
#---------------------------------------------------------------------
def sample_quadratic_bezier_curve(s, c, e, num_points=20, dtype='float32'):
    """
    Samples points from the quadratic bezier curves defined by the control points.
    Number of points to sample is num.

    Args:
        s (tensor): Start point of each curve, shape [N, 2].
        c (tensor): Control point of each curve, shape [N, 2].
        e (tensor): End point of each curve, shape [N, 2].
        num_points (int): Number of points to sample on every curve.

    Return:
       (tensor): Coordinates of the points on the Bezier curves, shape [N, num_points, 2] 
    """
    N, _ = s.shape.as_list()
    t = tf.linspace(0., 1., num_points)
    t = tf.cast(t, dtype=dtype)
    t = tf.stack([t] * N, axis=0)
    s_x = tf.expand_dims(s[..., 0], axis=1)
    s_y = tf.expand_dims(s[..., 1], axis=1)
    e_x = tf.expand_dims(e[..., 0], axis=1)
    e_y = tf.expand_dims(e[..., 1], axis=1)
    c_x = tf.expand_dims(c[..., 0], axis=1)
    c_y = tf.expand_dims(c[..., 1], axis=1)
    x = c_x + (1. - t) ** 2 * (s_x - c_x) + t ** 2 * (e_x - c_x)
    y = c_y + (1. - t) ** 2 * (s_y - c_y) + t ** 2 * (e_y - c_y)
    return tf.stack([x, y], axis=-1)


def renderer(curve_points, locations, colors, widths, H, W, K, canvas_color='gray', dtype='float32'):
    """                                                                                                  
    Renders the given brushstroke parameters onto a canvas.
    See Alg. 1 in https://arxiv.org/pdf/2103.17185.pdf.                                                  
                                                    
    Args:                                                                                                
        curve_points (tensor): Points specifying the curves that will be rendered on the canvas, shape [N, S, 2].
        locations (tensor): Location of each curve, shape [N, 2]. 
        colors (tensor): Color of each curve, shape [N, 3].
        widths (tensor): Width of each curve, shape [N, 1].
        H (int): Height of the canvas.              
        W (int): Width of the canvas.
        K (int): Number of brushstrokes to consider for each pixel, see Sec. C.2 of the paper (Arxiv version).
        canvas_color (str): Background color of the canvas. Options: 'gray', 'white', 'black', 'noise'.
    Returns:                                                                                             
        (tensor): The rendered canvas, shape [H, W, 3].                        
    """
    N, S, _ = curve_points.shape.as_list()
    # define coarse grid cell
    t_H = tf.linspace(0., float(H), int(H // 5))
    t_W = tf.linspace(0., float(W), int(W // 5))
    t_H = tf.cast(t_H, dtype=dtype)
    t_W = tf.cast(t_W, dtype=dtype)
    P_y, P_x = tf.meshgrid(t_W, t_H)
    P = tf.stack([P_x, P_y], axis=-1) # [32, 32, 2]
    # Compute now distances from every brushtroke center to every coarse grid cell
    #P_norms = tf.square(norm(P, axis=-1))
    #B_center_norms = tf.square(norm(locations, axis=-1))
    #P_dot_B_center = tf.einsum('xyf,Nf->xyN', P, locations)
    # [32, 32, N]
    #D_to_all_B_centers = tf.expand_dims(P_norms, axis=-1) + tf.expand_dims(tf.expand_dims(B_center_norms, axis=0), axis=0) - 2. * P_dot_B_center

    #####
    D_to_all_B_centers = tf.reduce_sum(tf.square(tf.expand_dims(P, axis=-2) - locations), axis=-1) # [H // C, W // C, N]
    #####

    # Find nearest brushstrokes' indices for every coarse grid cell
    _, idcs = tf.math.top_k(-D_to_all_B_centers, k=K) # [32, 32, K]
    # Now create 2 tensors (spatial size of a grid cell). One containing brushstroke locations, another containing
    # brushstroke colors.
    # [H // 10, W // 10, K, S, 2]
    canvas_with_nearest_Bs = tf.gather(params=curve_points,
                                       indices=idcs,
                                       batch_dims=0)
    # [H // 10, W // 10, K, 3]
    canvas_with_nearest_Bs_colors = tf.gather(params=colors,
                                              indices=idcs,
                                              batch_dims=0)
    # [H // 10, W // 10, K, 1]
    canvas_with_nearest_Bs_bs = tf.gather(params=widths,
                                          indices=idcs,
                                          batch_dims=0)
    # Resize those tensors to the full canvas size (not coarse grid)
    # First locations of points sampled from curves
    H_, W_, r1, r2, r3 = canvas_with_nearest_Bs.shape.as_list()
    canvas_with_nearest_Bs = tf.reshape(canvas_with_nearest_Bs, shape=(1, H_, W_, r1 * r2 * r3)) # [1, H // 10, W // 10, K * S * 2]
    canvas_with_nearest_Bs = tf.image.resize_nearest_neighbor(canvas_with_nearest_Bs, size=(H, W)) # [1, H, W, K * S * 2]
    canvas_with_nearest_Bs = tf.reshape(canvas_with_nearest_Bs, shape=(H, W, r1, r2, r3)) # [H, W, N, S, 2]
    # Now colors of curves
    H_, W_, r1, r2 = canvas_with_nearest_Bs_colors.shape.as_list()
    canvas_with_nearest_Bs_colors = tf.reshape(canvas_with_nearest_Bs_colors, shape=(1, H_, W_, r1 * r2)) # [1, H // 10, W // 10, K * 3]
    canvas_with_nearest_Bs_colors = tf.image.resize_nearest_neighbor(canvas_with_nearest_Bs_colors, size=(H, W)) # [1, H, W, K * 3]
    canvas_with_nearest_Bs_colors = tf.reshape(canvas_with_nearest_Bs_colors, shape=(H, W, r1, r2)) # [H, W, K, 3]
    # And with the brush size
    H_, W_, r1, r2 = canvas_with_nearest_Bs_bs.shape.as_list()
    canvas_with_nearest_Bs_bs = tf.reshape(canvas_with_nearest_Bs_bs, shape=(1, H_, W_, r1 * r2)) # [1, H // 10, W // 10, K]
    canvas_with_nearest_Bs_bs = tf.image.resize_nearest_neighbor(canvas_with_nearest_Bs_bs, size=(H, W)) # [1, H, W, K]
    canvas_with_nearest_Bs_bs = tf.reshape(canvas_with_nearest_Bs_bs, shape=(H, W, r1, r2)) # [H, W, K, 1]
    # Now create full-size canvas
    t_H = tf.linspace(0., float(H), H)
    t_W = tf.linspace(0., float(W), W)
    t_H = tf.cast(t_H, dtype=dtype)
    t_W = tf.cast(t_W, dtype=dtype)
    P_y, P_x = tf.meshgrid(t_W, t_H)
    P_full = tf.stack([P_x, P_y], axis=-1) # [H, W, 2]
    # Compute distance from every pixel on canvas to each (among nearest ones) line segment between points from curves
    canvas_with_nearest_Bs_a = tf.gather(canvas_with_nearest_Bs, axis=-2, indices=[i for i in range(S - 1)]) # start points of each line segment
    canvas_with_nearest_Bs_b = tf.gather(canvas_with_nearest_Bs, axis=-2, indices=[i for i in range(1, S)]) # end points of each line segments
    canvas_with_nearest_Bs_b_a = canvas_with_nearest_Bs_b - canvas_with_nearest_Bs_a # [H, W, N, S - 1, 2]
    P_full_canvas_with_nearest_Bs_a = tf.expand_dims(tf.expand_dims(P_full, axis=2), axis=2) - canvas_with_nearest_Bs_a # [H, W, K, S - 1, 2]
    # compute t value for which each pixel is closest to each line that goes through each line segment (among nearest ones)
    t = tf.reduce_sum(canvas_with_nearest_Bs_b_a * P_full_canvas_with_nearest_Bs_a, axis=-1) \
        / (tf.reduce_sum(tf.square(canvas_with_nearest_Bs_b_a), axis=-1) + 1e-8)
    # if t value is outside [0, 1], then the nearest point on the line does not lie on the segment, so clip values of t
    t = tf.clip_by_value(t, clip_value_min=0.0, clip_value_max=1.0)
    # compute closest points on each line segment - [H, W, K, S - 1, 2]
    closest_points_on_each_line_segment = canvas_with_nearest_Bs_a + tf.expand_dims(t, axis=-1) * canvas_with_nearest_Bs_b_a
    # compute the distance from every pixel to the closest point on each line segment - [H, W, K, S - 1]
    dist_to_closest_point_on_line_segment = \
        tf.reduce_sum(tf.square(tf.expand_dims(tf.expand_dims(P_full, axis=2), axis=2) - closest_points_on_each_line_segment), axis=-1)
    # and distance to the nearest bezier curve.
    D = tf.reduce_min(dist_to_closest_point_on_line_segment, axis=[-1, -2]) # [H, W]
    # Finally render curves on a canvas to obtain image.
    I_NNs_B_ranking = tf.nn.softmax(100000. * (1.0 / (1e-8 + tf.reduce_min(dist_to_closest_point_on_line_segment, axis=[-1]))), axis=-1) # [H, W, N]
    I_colors = tf.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_colors, I_NNs_B_ranking) # [H, W, 3]
    bs = tf.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_bs, I_NNs_B_ranking) # [H, W, 1]
    bs_mask = tf.math.sigmoid(bs - tf.expand_dims(D, axis=-1))
    if canvas_color == 'gray':
        canvas = tf.ones(shape=I_colors.shape, dtype=dtype) * 0.5
    elif canvas_color == 'white':
        canvas = tf.ones(shape=I_colors.shape, dtype=dtype)
    elif canvas_color == 'black':
        canvas = tf.zeros(shape=I_colors.shape, dtype=dtype)
    elif canvas_color == 'noise':
        canvas = tf.random.normal(shape=I_colors.shape, dtype=dtype) * 0.1

    I = I_colors * bs_mask + (1 - bs_mask) * canvas
    return I


#---------------------------------------------------------------------
# Losses
#---------------------------------------------------------------------
def content_loss(features_lhs, features_rhs, layers, weights, scale_by_y=False):
    """
    Computes the VGG perceptual loss.
    
    Args:
        features_lhs (dict of tensors): Dictionary of VGG activations.
        features_rhs (dict of tensors): Dictionary of VGG activations.
        layers (list of str): List specifying the layers to use.
        weights (list of floats): List specifying the weights for the used layers.

    Returns:
        VGG perceptual loss.
    """

    feat_lhs = [features_lhs[key] for key in layers]
    feat_rhs = [features_rhs[key] for key in layers]

    if scale_by_y:
        losses = [w * tf.reduce_mean(tf.square(xf - yf) * tf.minimum(yf, tf.sigmoid(yf))) for w, xf, yf in zip(weights, feat_lhs, feat_rhs)]
    else:
        losses = [w * tf.reduce_mean(tf.square(xf - yf)) for w, xf, yf in zip(weights, feat_lhs, feat_rhs)] 

    loss = tf.add_n(losses)
    return loss


def get_gram_matrices(features):
    """
    Computes the gram matrices for the given list of activations. 

    Args:
        features (list of tensors): Dictionary of VGG activations.
    
    Returns:
        List of gram matrices.
    """
    gram_matrices = []
    for feature in features:
        gram_matrix = tf.einsum('bhwf,bhwl->bfl', feature, feature)
        B, H, W, C = feature.shape.as_list()
        gram_matrix /= tf.cast(H * W * C, dtype=tf.float32)
        gram_matrices.append(gram_matrix)
    return gram_matrices


def style_loss(features_lhs, features_rhs, layers, weights):
    """
    Computes the VGG gram matrix style loss.
    
    Args:
        features_lhs (dict of tensors): Dictionary of VGG activations.
        features_rhs (dict of tensors): Dictionary of VGG activations.
        layers (list of str): List specifying the layers to use.
        weights (list of floats): List specifying the weights for the used layers.

    Returns:
        VGG gram matrix style loss.
    """
    feat_lhs = [features_lhs[key] for key in layers]
    feat_rhs = [features_rhs[key] for key in layers]
    gram_matrices_lhs = get_gram_matrices(feat_lhs)
    gram_matrices_rhs = get_gram_matrices(feat_rhs)
    losses = [w * tf.reduce_sum(tf.square(gram_lhs - gram_rhs)) for w, gram_lhs, gram_rhs in zip(weights, gram_matrices_lhs, gram_matrices_rhs)]
    loss = tf.add_n(losses)
    return loss


def get_nn_idxs(X, k, fetch_dist=False):                                                                       
    """
    For a given tensor compute all the nearest neighbor indices to each element.                                                                        

    Args:
        x (tensor): Tensor of shape [B, N, F].
        k (int): Number of nearest neighbors.
        fetch_dist (bool): Also return the distances.

    Returns:
        Tensor of shape [B, N, k].
        
    """                                                                                                                                 
    r = tf.reduce_sum(X * X, 2, keepdims=True)                                                                                      
    D = r - 2 * tf.matmul(X, tf.transpose(X, perm=(0, 2, 1))) + tf.transpose(r, perm=(0, 2, 1))                                                          
    X_top_vals, X_top_idxs = tf.math.top_k(-D, k=k, sorted=True, name=None)

    if fetch_dist:                                                                                                                      
        return X_top_idxs, X_top_vals
    else:
        return X_top_idxs


def total_variation_loss(x_loc, s, e, K=10):

    def projection(z):
        x = tf.gather(z, axis=-1, indices=[0])
        y = tf.gather(z, axis=-1, indices=[1])
        return tf.concat([tf.square(x), tf.square(y), x * y], axis=-1) 

    se_vec = e - s
    se_vec_proj = projection(se_vec)
    
    x_nn_idcs = get_nn_idxs(tf.expand_dims(x_loc, axis=0), k=K)

    x_nn_idcs = tf.squeeze(x_nn_idcs, axis=0)
    x_sig_nns = tf.gather(se_vec, indices=x_nn_idcs, axis=0, batch_dims=0)
    
    dist_to_centroid = tf.reduce_mean(tf.reduce_sum(tf.square(projection(x_sig_nns) - tf.expand_dims(projection(se_vec), axis=-2)), axis=-1))
    return dist_to_centroid 


def draw_projection_loss(location, s, e, draw_curve_position, draw_curve_vector, draw_strength):
    dist = tf.reduce_sum(tf.square(tf.expand_dims(draw_curve_position, axis=1) - location), axis=-1)
    _, idcs = tf.math.top_k(-dist, k=draw_strength) # [num_points, K]
    se_vec = e - s
    strokes_vec_nn = tf.gather(se_vec, indices=idcs, axis=0) # [num_points, K, 2]
    strokes_vec_nn /= (norm(strokes_vec_nn, axis=-1, keepdims=True) + 1e-6)
    curves_vec = draw_curve_vector / (norm(draw_curve_vector, axis=-1, keepdims=True) + 1e-6) 
    projection = tf.abs(tf.einsum('mki,mi->mk', strokes_vec_nn, curves_vec)) # [num_points, num_strokes]
    projection_loss = tf.reduce_mean(tf.square(1 - projection))
    return projection_loss


def curviture_loss(s, e, c):
    v1 = s - c
    v2 = e - c
    dist_se = norm(e - s, axis=-1) + 1e-6
    return tf.reduce_mean(norm(v1 + v2, axis=-1) / dist_se)


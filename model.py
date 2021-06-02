import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf

import os
import numpy as np
from PIL import Image
from tqdm import trange

import networks
import ops
import utils


def stylize(content_img,
            style_img,
            # Brushstroke optimizer params
            resolution=512,
            num_strokes=5000,
            num_steps=100,
            S=10,
            K=20,
            canvas_color='gray',
            width_scale=0.1,
            length_scale=1.1,
            content_weight=1.0,
            style_weight=3.0,
            tv_weight=0.008,
            curviture_weight=4.0,
            # Pixel optimizer params
            pixel_resolution=1024,
            num_steps_pixel=2000
            ):

    stroke_optim = BrushstrokeOptimizer(content_img,
                                        style_img,
                                        resolution=resolution,
                                        num_strokes=num_strokes,
                                        num_steps=num_steps,
                                        S=S,
                                        K=K,
                                        canvas_color=canvas_color,
                                        width_scale=width_scale,
                                        length_scale=length_scale,
                                        content_weight=content_weight,
                                        style_weight=style_weight,
                                        tv_weight=tv_weight,
                                        curviture_weight=curviture_weight)
    print('Stroke optimization:')
    canvas = stroke_optim.optimize()

    pixel_optim = PixelOptimizer(canvas,
                                 style_img,
                                 resolution=pixel_resolution,
                                 num_steps=num_steps_pixel,
                                 content_weight=1.0,
                                 style_weight=10000.0)

    print('Pixel optimization:')
    canvas = pixel_optim.optimize()
    return canvas


class BrushstrokeOptimizer:

    def __init__(self,
                 content_img,                              # Content image (PIL.Image).
                 style_img,                                # Style image (PIL.Image).
                 draw_curve_position_path = None,          # Set of points that represent the drawn curves, denoted as P_i in Sec. B of the paper (str).
                 draw_curve_vector_path   = None,          # Set of tangent vectors for the points of the drawn curves, denoted as v_i in Sec. B of the paper (str).
                 draw_strength            = 100,           # Strength of the influence of the drawn curves, denoted L in Sec. B of the paper (int).
                 resolution               = 512,           # Resolution of the canvas (int).
                 num_strokes              = 5000,          # Number of brushstrokes (int).
                 num_steps                = 100,           # Number of optimization steps (int).
                 S                        = 10,            # Number of points to sample on each curve, see Sec. 4.2.1 of the paper (int).
                 K                        = 20,            # Number of brushstrokes to consider for each pixel, see Sec. C.2 of the paper (int).
                 canvas_color             = 'gray',        # Color of the canvas (str).
                 width_scale              = 0.1,           # Scale parameter for the brushstroke width (float).
                 length_scale             = 1.1,           # Scale parameter for the brushstroke length (float).
                 content_weight           = 1.0,           # Weight for the content loss (float).
                 style_weight             = 3.0,           # Weight for the style loss (float).
                 tv_weight                = 0.008,         # Weight for the total variation loss (float).
                 draw_weight              = 100.0,         # Weight for the drawing projection loss (float)
                 curviture_weight         = 4.0,           # Weight for the curviture loss (float).
                 streamlit_pbar           = None,          # Progressbar for streamlit app (obj).
                 dtype                    = 'float32'      # Data type (str).
                ):
    
        #content_img = Image.open(f'/export/home/mwright/data/neural_painter/image_pairs/photos/{self.args.content_img}')
        #style_img = Image.open(f'/export/home/mwright/data/neural_painter/image_pairs/artworks/{self.args.style_img}')

        self.draw_strength = draw_strength
        self.draw_weight = draw_weight
        self.resolution = resolution
        self.num_strokes = num_strokes
        self.num_steps = num_steps
        self.S = S
        self.K = K
        self.canvas_color = canvas_color
        self.width_scale = width_scale
        self.length_scale = length_scale
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.curviture_weight = curviture_weight
        self.streamlit_pbar = streamlit_pbar
        self.dtype = dtype

        # Set canvas size (set smaller side of content image to 'resolution' and scale other side accordingly)
        W, H = content_img.size
        if H < W:                                                                                        
            new_H = resolution                                                                           
            new_W = int((W / H) * new_H)
        else:                                                                                            
            new_W = resolution
            new_H = int((H / W) * new_W)                                                                 
                                                                                                         
        self.canvas_height = new_H
        self.canvas_width = new_W

        content_img = content_img.resize((self.canvas_width, self.canvas_height))
        style_img = style_img.resize((self.canvas_width, self.canvas_height))
        
        content_img = np.array(content_img).astype(self.dtype)
        style_img = np.array(style_img).astype(self.dtype)

        content_img /= 255.0
        style_img /= 255.0

        self.content_img_np = content_img
        self.style_img_np = style_img

        if draw_curve_position_path is not None and draw_curve_vector_path is not None:
            self.draw_curve_position_np = np.load(draw_curve_position_path)
            self.draw_curve_vector_np = np.load(draw_curve_vector_path)
            self.draw_curve_position_np[..., 0] *= self.canvas_width
            self.draw_curve_position_np[..., 1] *= self.canvas_height

        ckpt_path = utils.download_weights(url='https://www.dropbox.com/s/hv7b4eajrj7isyq/vgg_weights.pickle?dl=1',
                                           name='vgg_weights.pickle')
        self.vgg = networks.VGG(ckpt_path=ckpt_path)

    def optimize(self):
        self._initialize()

        steps = trange(self.num_steps, desc='', leave=True)
        for step in steps:
            self._optimize()
            steps.set_description(f'content_loss: {self.loss_dict["content"].numpy().item():.6f},\
                                  style_loss: {self.loss_dict["style"].numpy().item():.6f}')

            if self.streamlit_pbar is not None: self.streamlit_pbar.update(1)
        return Image.fromarray(np.array(np.clip(self._render().numpy(), 0, 1) * 255, dtype=np.uint8))

    def _initialize(self):
        location, s, e, c, width, color = utils.initialize_brushstrokes(self.content_img_np, 
                                                                        self.num_strokes, 
                                                                        self.canvas_height, 
                                                                        self.canvas_width, 
                                                                        self.length_scale, 
                                                                        self.width_scale)

        self.curve_s = tf.Variable(name='curve_s', initial_value=s, dtype=self.dtype)
        self.curve_e = tf.Variable(name='curve_e', initial_value=e, dtype=self.dtype)
        self.curve_c = tf.Variable(name='curve_c', initial_value=c, dtype=self.dtype)
        self.color = tf.Variable(name='color', initial_value=color, dtype=self.dtype)
        self.location = tf.Variable(name='location', initial_value=location, dtype=self.dtype)
        self.width = tf.Variable(name='width', initial_value=width, dtype=self.dtype)
        self.content_img = tf.constant(name='content_img', value=self.content_img_np, dtype=self.dtype)
        self.style_img = tf.constant(name='style_img', value=self.style_img_np, dtype=self.dtype)

        if hasattr(self, 'draw_curve_position_np') and hasattr(self, 'draw_curve_vector_np'):
            self.draw_curve_position = tf.constant(name='draw_curve_position', value=self.draw_curve_position_np, dtype=self.dtype)
            self.draw_curve_vector = tf.constant(name='draw_curve_vector', value=self.draw_curve_vector_np, dtype=self.dtype)

        self.params_dict = {'location': self.location, 
                            'curve_s': self.curve_s, 
                            'curve_e': self.curve_e, 
                            'curve_c': self.curve_c, 
                            'width': self.width, 
                            'color': self.color}

    def _render(self):
        curve_points = ops.sample_quadratic_bezier_curve(s=self.curve_s + self.location,
                                                         e=self.curve_e + self.location,
                                                         c=self.curve_c + self.location,
                                                         num_points=self.S,
                                                         dtype=self.dtype)

        I = ops.renderer(curve_points, 
                         self.location, 
                         self.color, 
                         self.width, 
                         self.canvas_height, 
                         self.canvas_width, 
                         self.K, 
                         canvas_color=self.canvas_color, 
                         dtype=self.dtype)

        return I

    def _optimize(self):
        # resize images to save memory

        def loss():
            I = self._render()
            rendered_canvas_resized = \
                tf.image.resize(images=ops.preprocess_img(I),
                                size=(int(self.canvas_height // 2), int(self.canvas_width // 2)),
                                method='nearest')
            content_img_resized = \
                tf.image.resize(images=ops.preprocess_img(self.content_img),
                                size=(int(self.canvas_height // 2), int(self.canvas_width // 2)),
                                method='nearest')
            style_img_resized = \
                tf.image.resize(images=ops.preprocess_img(self.style_img),
                                size=(int(self.canvas_height // 2), int(self.canvas_width // 2)),
                                method='nearest')
    
            self.loss_dict = {}
            self.loss_dict['content'] = ops.content_loss(self.vgg.extract_features(rendered_canvas_resized),
                                                         self.vgg.extract_features(content_img_resized),
                                                         #layers=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2'],
                                                         layers=['conv4_2', 'conv5_2'],
                                                         weights=[1, 1],
                                                         scale_by_y=True)
            self.loss_dict['content'] *= self.content_weight

            self.loss_dict['style'] = ops.style_loss(self.vgg.extract_features(rendered_canvas_resized),
                                                     self.vgg.extract_features(style_img_resized),
                                                     layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                                     weights=[1, 1, 1, 1, 1])
            self.loss_dict['style'] *= self.style_weight

            self.loss_dict['tv'] = ops.total_variation_loss(x_loc=self.location, s=self.curve_s, e=self.curve_e, K=10)
            self.loss_dict['tv'] *= self.tv_weight

            if hasattr(self, 'draw_curve_position') and hasattr(self, 'draw_curve_vector'):
                self.loss_dict['drawing'] = ops.draw_projection_loss(self.location, 
                                                                     self.curve_s, 
                                                                     self.curve_e, 
                                                                     self.draw_curve_position, 
                                                                     self.draw_curve_vector, 
                                                                     self.draw_strength)
                self.loss_dict['drawing'] *= self.draw_weight

            self.loss_dict['curviture'] = ops.curviture_loss(self.curve_s, self.curve_e, self.curve_c)
            self.loss_dict['curviture'] *= self.curviture_weight
            loss = tf.constant(0.0)
            for key in self.loss_dict:
                loss += self.loss_dict[key]
            return loss

        def style_loss():
            I = self._render()
            rendered_canvas_resized = \
                tf.image.resize(images=ops.preprocess_img(I),
                                size=(int(self.canvas_height // 2), int(self.canvas_width // 2)),
                                method='bilinear')
            content_img_resized = \
                tf.image.resize(images=ops.preprocess_img(self.content_img),
                                size=(int(self.canvas_height // 2), int(self.canvas_width // 2)),
                                method='bilinear')
            style_img_resized = \
                tf.image.resize(images=ops.preprocess_img(self.style_img),
                                size=(int(self.canvas_height // 2), int(self.canvas_width // 2)),
                                method='bilinear')

            loss = ops.style_loss(self.vgg.extract_features(rendered_canvas_resized),
                                  self.vgg.extract_features(style_img_resized),
                                  layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                  weights=[1, 1, 1, 1, 1])
            loss *= self.style_weight
            return loss

        tf.keras.optimizers.Adam(learning_rate=0.1).minimize(loss, var_list=[self.location, self.curve_s, self.curve_e, self.curve_c, self.width])
        tf.keras.optimizers.Adam(learning_rate=0.01).minimize(style_loss, var_list=[self.color])
        self._constraints()

    def _constraints(self):
        loc_x, loc_y = tf.gather(self.location, axis=-1, indices=[0]), tf.gather(self.location, axis=-1, indices=[1])
        loc_clip = tf.concat([tf.clip_by_value(loc_x, 0, self.canvas_height), tf.clip_by_value(loc_y, 0, self.canvas_width)], axis=-1)
        self.location.assign(loc_clip)
        self.color.assign(tf.clip_by_value(self.color, 0, 1))
        self.width.assign(tf.nn.relu(self.width))


class PixelOptimizer:

    def __init__(self,
                 canvas,                              # Canvas (PIL.Image).
                 style_img,                           # Style image (PIL.Image).
                 resolution          = 1024,          # Resolution of the canvas.
                 num_steps           = 5000,          # Number of optimization steps.
                 content_weight      = 1.0,           # Weight for the content loss.
                 style_weight        = 10000.0,       # Weight for the style loss.
                 tv_weight           = 0.0,           # Weight for the total variation loss.
                 streamlit_pbar      = None,          # Progressbar for streamlit app (obj).
                 dtype               = 'float32'      # Data type.
                ):
    
        self.resolution = resolution
        self.num_steps = num_steps
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.streamlit_pbar = streamlit_pbar
        self.dtype = dtype

        # Set canvas size (set smaller side of content image to 'resolution' and scale other side accordingly)
        W, H = canvas.size
        if H < W:                                                                                        
            new_H = resolution                                                                           
            new_W = int((W / H) * new_H)
        else:                                                                                            
            new_W = resolution
            new_H = int((H / W) * new_W)                                                                 
                                                                                                         
        self.canvas_height = new_H              
        self.canvas_width = new_W

        canvas = canvas.resize((self.canvas_width, self.canvas_height))
        style_img = style_img.resize((self.canvas_width, self.canvas_height))
        
        canvas = np.array(canvas).astype(self.dtype)
        style_img = np.array(style_img).astype(self.dtype)

        canvas /= 255.0
        style_img /= 255.0

        content_img = canvas

        self.canvas_np = canvas
        self.content_img_np = canvas
        self.style_img_np = style_img

        ckpt_path = utils.download_weights(url='https://www.dropbox.com/s/hv7b4eajrj7isyq/vgg_weights.pickle?dl=1',
                                           name='vgg_weights.pickle')
        self.vgg = networks.VGG(ckpt_path=ckpt_path)

    def optimize(self):
        self._initialize()

        steps = trange(self.num_steps, desc='', leave=True)
        for step in steps:
            self._optimize()

            s = ''
            for key in self.loss_dict:
                loss = self.loss_dict[key].numpy().item()
                s += key + f': {loss:.4f}, '

            steps.set_description(s[:-2]) 
            steps.refresh()
            if self.streamlit_pbar is not None: self.streamlit_pbar.update(1)
        return Image.fromarray(np.array(np.clip(self.canvas.numpy(), 0, 1) * 255, dtype=np.uint8))

    def _initialize(self):
        self.canvas = tf.Variable(name='canvas', initial_value=self.canvas_np, dtype=self.dtype)
        self.content_img = tf.constant(name='content_img', value=self.content_img_np, dtype=self.dtype)
        self.style_img = tf.constant(name='style_img', value=self.style_img_np, dtype=self.dtype)

    def _optimize(self):
        
        def loss():
            factor = 1
            resize_method = 'nearest'

            # resize images to save memory
            rendered_canvas_resized = \
                tf.image.resize(images=ops.preprocess_img(self.canvas),
                                size=(int(self.canvas_height * factor), int(self.canvas_width * factor)),
                                method=resize_method)

            content_img_resized = \
                tf.image.resize(images=ops.preprocess_img(self.content_img),
                                size=(int(self.canvas_height * factor), int(self.canvas_width * factor)),
                                method=resize_method)

            style_img_resized = \
                tf.image.resize(images=ops.preprocess_img(self.style_img),
                                size=(int(self.canvas_height * factor), int(self.canvas_width * factor)),
                                method=resize_method)

            self.loss_dict = {}
            self.loss_dict['content'] = ops.content_loss(self.vgg.extract_features(rendered_canvas_resized),
                                                    self.vgg.extract_features(content_img_resized),
                                                    layers=['conv1_2_pool', 'conv2_2_pool', 'conv3_3_pool', 'conv4_3_pool', 'conv5_3_pool'],
                                                    weights=[1, 1, 1, 1, 1])
            self.loss_dict['content'] *= self.content_weight

            self.loss_dict['style'] = ops.style_loss(self.vgg.extract_features(rendered_canvas_resized),
                                                self.vgg.extract_features(style_img_resized),
                                                layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                                weights=[1, 1, 1, 1, 1])
            self.loss_dict['style'] *= self.style_weight

            self.loss_dict['tv'] = ((tf.nn.l2_loss(self.canvas[1:, :, :] - self.canvas[:-1, :, :]) / self.canvas.shape.as_list()[0]) +
                               (tf.nn.l2_loss(self.canvas[:, 1:, :] - self.canvas[:, :-1, :]) / self.canvas.shape.as_list()[1]))
            self.loss_dict['tv'] *= self.tv_weight

            loss = tf.constant(0.0)
            for key in self.loss_dict:
                loss += self.loss_dict[key]
            return loss

        tf.keras.optimizers.Adam(learning_rate=0.01).minimize(loss, var_list=[self.canvas])
        self._constraints()

    def _constraints(self):
        self.canvas.assign(tf.clip_by_value(self.canvas, 0, 1))




import numpy as np
from PIL import Image
from typing import Sequence
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from stqdm import stqdm
import os
import base64

from model import BrushstrokeOptimizer, PixelOptimizer


def parse_paths(json_obj, height, width):
    xs = []
    ys = []
    for segments in json_obj['path']:
        if segments[0] == 'Q':
            xs.append(segments[2] / width)
            xs.append(segments[4] / width)
            ys.append(segments[1] / height)
            ys.append(segments[3] / height)
    xs = np.array(xs)
    ys = np.array(ys)
    return np.stack((xs, ys), axis=1)


def sample_vectors(points, lookahead=10, freq=10):
    idcs = np.arange(points.shape[0])[::freq]
    vectors = []
    positions = []
    for i in range(idcs.shape[0] - lookahead):
        vectors.append(points[idcs[i] + lookahead] - points[idcs[i]])
        positions.append(points[idcs[i]])
    return np.array(vectors), np.array(positions)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    # Taken from: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/27
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def resize(img, size, interpolation=Image.BILINEAR):
    # https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


st.sidebar.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
.medium-font {
    font-size:15px !important;
}
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
## Content and Style image
st.sidebar.markdown('<p class="medium-font"><b>Content Image</b></p>', unsafe_allow_html=True)
selected_content_image = st.sidebar.selectbox('Select content image:', [None] + os.listdir('images/content'))
st.sidebar.text('OR')
uploaded_content_image = st.sidebar.file_uploader('Upload content image:', type=['png', 'jpg'])

st.sidebar.markdown('<p class="medium-font"><b>Style Image</b></p>', unsafe_allow_html=True)
selected_style_image = st.sidebar.selectbox('Select style image:', [None] + os.listdir('images/style'))
st.sidebar.text('OR')
uploaded_style_image = st.sidebar.file_uploader('Upload style image:', type=['png', 'jpg'])

## Parameters
st.sidebar.markdown('<p class="medium-font"><b>Options</b></p>', unsafe_allow_html=True)
num_steps_stroke = st.sidebar.slider('Brushstroke optimization steps:', 20, 100, 100)
num_steps_pixel = st.sidebar.slider('Pixel optimization steps:', 100, 5000, 2000)
content_weight = st.sidebar.slider('Content weight:', 1.0, 50.0, 1.0)
style_weight = st.sidebar.slider('Style weight:', 1.0, 50.0, 3.0)
draw_weight = st.sidebar.slider('Drawing weight', 50.0, 200.0, 100.0)
draw_strength = st.sidebar.slider('Drawing strength (denoted L in the paper):', 50, 200, 100)
stroke_width = st.sidebar.slider('Stroke width:', 0.01, 2.0, 0.1)
stroke_length = st.sidebar.slider('Stroke length:', 0.1, 2.0, 1.1)


#drawing_mode = st.sidebar.selectbox(
#    'Drawing tool:', ('freedraw', 'line', 'rect', 'circle', 'transform')
#)
realtime_update = st.sidebar.checkbox('Update in realtime', True)

# Main
stroke_color = st.color_picker('Stroke color hex: ', '#ff0000')


content_img = None
if selected_content_image is not None: content_img = Image.open(os.path.join('images/content', selected_content_image))
if uploaded_content_image is not None: content_img = Image.open(uploaded_content_image)

style_img = None
if selected_style_image is not None: style_img = Image.open(os.path.join('images/style', selected_style_image))
if uploaded_style_image is not None: style_img = Image.open(uploaded_style_image)

if content_img is None or style_img is None:
    st.image(Image.open('docs/img/left_arrow.png'))
    st.image(Image.open('docs/img/down_arrow.png'))
    #st.markdown('<p class="medium-font"><b>Select or upload content and style images...</b></p>', unsafe_allow_html=True)


col1, col2 = st.beta_columns(2)

# Preview images
if content_img is not None:
    content_thumb = resize(content_img, size=400)
    col1.header('Content image')
    col1.image(content_img, use_column_width=True)
if style_img is not None:
    style_thumb = resize(style_img, size=400)
    col2.header('Style image')
    col2.image(style_thumb, use_column_width=True)


if content_img is not None and style_img is not None:
    if not os.path.exists('.temp'):
        os.makedirs('.temp')

    content_img_name = content_img.filename
    content_img = content_img.convert('RGB')
    content_img.save(f'.temp/content_img.jpg')
    style_img_name = style_img.filename
    style_img = style_img.convert('RGB')
    style_img.save(f'.temp/style_img.jpg')

    height = content_img.size[1]
    width = content_img.size[0]
    factor = 1.0
    # resize image such that the largest side is 512 because else the canvas drawer messes up
    if width > 512 or height > 512:
        if width < height:
            height = int(512 * (height / width))
            width = 512
            factor *= height / width
        else:
            width = int(512 * (width / height))
            height = 512
            factor *= width / height

    st.text('Now draw some curves on the canvas.')
    st.text('To draw a curve:')
    st.text('- hold down the left mouse button')
    st.text('- and slowly move the mouse over the canvas.')

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color='rgba(255, 165, 0, 0.3)',  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color=stroke_color,
        background_color='' if content_img else '#eee',
        background_image=content_img,
        update_streamlit=realtime_update,
        height=height,
        width=width,
        drawing_mode='freedraw',
        #key='canvas',
    )

    if canvas_result.json_data is not None:
        if len(canvas_result.json_data['objects']) > 0:
            if st.button('Stylize'):
                vectors_all = []
                positions_all = []
                img_array = np.array(content_img)
                for i in range(len(canvas_result.json_data['objects'])):
                    points = parse_paths(canvas_result.json_data['objects'][i], float(height), float(width))
                    vectors, positions = sample_vectors(points, lookahead=5, freq=5)
                    vectors_all.append(vectors)
                    positions_all.append(positions)

                    for i in range(points.shape[0]):
                        y = int(points[i, 0] * content_img.size[0])
                        x = int(points[i, 1] * content_img.size[1])
                        img_array[y-2:y+2, x-2:x+2] = np.array([255, 0, 0])

                vectors_all = np.concatenate(vectors_all, axis=0).astype(np.float32)
                positions_all = np.concatenate(positions_all, axis=0).astype(np.float32)
                np.save('.temp/vectors', vectors_all)
                np.save('.temp/positions', positions_all)
                
                content_img = Image.open('.temp/content_img.jpg')
                style_img = Image.open('.temp/style_img.jpg')

                st.text('Brushstroke optimization...')
                pbar = stqdm(range(num_steps_stroke))
                stroke_optim = BrushstrokeOptimizer(content_img, 
                                                    style_img, 
                                                    draw_curve_position_path='.temp/positions.npy',
                                                    draw_curve_vector_path='.temp/vectors.npy',
                                                    draw_strength=draw_strength,
                                                    resolution=512,
                                                    num_strokes=5000,
                                                    num_steps=num_steps_stroke,
                                                    width_scale=stroke_width,
                                                    length_scale=stroke_length,
                                                    content_weight=content_weight,
                                                    style_weight=style_weight,
                                                    draw_weight=draw_weight,
                                                    streamlit_pbar=pbar)
                canvas = stroke_optim.optimize()

                st.text('Pixel optimization...')
                pbar = stqdm(range(num_steps_pixel))
                pixel_optim = PixelOptimizer(canvas,
                                             style_img,
                                             resolution=1024,
                                             num_steps=num_steps_pixel,
                                             content_weight=1.0,
                                             style_weight=10000.0,
                                             streamlit_pbar=pbar)
                canvas = pixel_optim.optimize()
                
                st.text('Stylized image:')
                st.image(canvas.resize((width, height)))

                canvas.save('.temp/canvas.jpg')
                st.markdown(get_binary_file_downloader_html('.temp/canvas.jpg', 'stylized image in high resolution'), unsafe_allow_html=True)
                
                

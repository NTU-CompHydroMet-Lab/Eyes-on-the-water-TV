from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64
import diskcache
import os
import multiprocessing
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from baseline_config import load_baselines, set_baseline, get_baseline

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

# from sam2.build_sam import build_sam2, build_sam2_video_predictor
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO, SAM
import torch
import pandas as pd
import numpy as np
np.seterr(over='ignore') # Ignore overflow errors


from dash import html, dcc, callback, long_callback
from dash.dependencies import Input, Output, State, ALL
from dash.long_callback import DiskcacheLongCallbackManager
import dash
import plotly.graph_objects as go
import webcolors
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


# Add at the top of your file, after imports
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    long_callback_manager=long_callback_manager,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Add custom CSS as an external stylesheet
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

image_parent_folder = 'sample_data'

# Add default settings
DEFAULT_SETTINGS = {
    'Object detection': {
        'confidence': 0.6,
        'show_overlay': True,
    },
    'Segmentation': {
        'show_overlay': True,
        'point_x': 0.5,  # Default center position (50%)
        'point_y': 0.6   # Default position at 60% height
    },
    'Water Clarity Index': {
        'show_score': True,
    }
}

# Initialize the AI models
def model_init():

    # Load the object detection model
    object_detection_model = YOLO("models/yolo11x.pt")

    # # Load the segmentation model
    sam2_segmentation_model = SAM("models/sam2.1_b.pt")
    # sam2_segmentation_model = SAM("models/sam2.1_t.pt")

    # # Load the water Clarity Index model
    water_clearity_index_model = YOLO("models/WCI_cls_best.pt")

    return {
        'object_detection': object_detection_model,
        'segmentation': sam2_segmentation_model,
        'water_clearity_index': water_clearity_index_model,
    }

# region Segmentation functions
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
# endregion

# Function to get the closest colour name
def closest_colour(requested_colour):
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

# Function to get the actual colour name
def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

# Function to get list of subfolders
def get_subfolders(parent_folder):
    return [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

# Function to get images from a folder
def get_images(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    return [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Add a simple image cache to store processed images
app.image_cache = {}

# Modify the process_image function to use the cache
def process_image(img, active_toggle, models, settings_store, img_path=None):
    if not active_toggle or not img_path:
        return img
    
    # Create cache key from image path and settings
    cache_key = f"{img_path}_{active_toggle}_{str(settings_store[active_toggle])}"
    
    # Check if image is in cache
    if cache_key in app.image_cache:
        return app.image_cache[cache_key]
    
    # Process image as before
    settings = settings_store[active_toggle]
    draw = ImageDraw.Draw(img)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 80)
    fnt = ImageFont.load_default(80)
    
    if active_toggle == 'Object detection':
        results = models['object_detection'].predict(
            img,
            conf=settings['confidence'],
            verbose=False,
        )[0]
        
        if settings['show_overlay']:
            img_np = results.plot()[:, :, ::-1]
            img = Image.fromarray(img_np)
        
    elif active_toggle == 'Segmentation':
        if settings['show_overlay']:
            # Get point settings
            point_x = settings['point_x']
            point_y = settings['point_y']
            
            # Standardize cache path construction
            cache_folder = os.path.dirname(img_path).replace(image_parent_folder, '').lstrip('/')
            seg_cache_folder = os.path.join(
                "cache", "segmentation",
                cache_folder,
                f"point_{point_x:.2f}_{point_y:.2f}"
            )
            os.makedirs(seg_cache_folder, exist_ok=True)

            # Update mask path
            mask_path = os.path.join(
                seg_cache_folder,
                os.path.basename(img_path).lower().replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')
            )
            
            if not os.path.exists(mask_path):
                # Calculate point position based on image size
                point_x_px = int(img.size[0] * point_x)
                point_y_px = int(img.size[1] * point_y)
                
                # print(f"Using point: ({point_x_px}, {point_y_px})")
                
                results = models['segmentation'].predict(
                    img,
                    points=[[point_x_px, point_y_px]],
                    verbose=False
                )
                single_mask = results[0].masks.data.cpu().numpy()[0]
                np.save(mask_path, single_mask)
                print(f"Saved mask to {mask_path}")
            
            single_mask = np.load(mask_path)
            
            # Convert PIL image to NumPy array
            img_np = np.array(img, dtype=np.uint8)

            # Define the orange color
            orange_color = np.array([255, 165, 0], dtype=np.uint8)  # RGB for orange

            # Create an overlay with the same size as the image
            overlay = np.zeros_like(img_np, dtype=np.uint8)

            # Apply the orange color to the regions specified by the mask
            overlay[single_mask == True] = orange_color

            # Add the orange overlay directly to the image
            alpha = 0.5  # Transparency level for the overlay
            img_np = img_np.astype(np.float32)  # Convert to float for blending
            overlay = overlay.astype(np.float32)

            # Increase intensity by adding the mask directly to the image
            blended = img_np * (1 - alpha) + overlay * alpha

            # Convert back to uint8 and PIL Image
            img = Image.fromarray(np.uint8(blended))
            
            # Draw the point marker
            draw = ImageDraw.Draw(img)
            point_x_px = int(img.size[0] * point_x)
            point_y_px = int(img.size[1] * point_y)
            
            # 1% of the image size
            marker_size = int(img.size[0] * 0.01)
            
            # Draw a cross marker with white outline
            draw.line((point_x_px - marker_size, point_y_px, point_x_px + marker_size, point_y_px), 
                     fill='green', width=5)
            draw.line((point_x_px, point_y_px - marker_size, point_x_px, point_y_px + marker_size), 
                     fill='green', width=5)
            
            # Draw a circle around the cross
            draw.ellipse((point_x_px - marker_size, point_y_px - marker_size, 
                         point_x_px + marker_size, point_y_px + marker_size), 
                        outline='white', width=4)
        
    elif active_toggle == 'Water Clarity Index':
        results = models['water_clearity_index'].predict(img, verbose=False)
        probs = results[0].probs.data.cpu().numpy()
        overall_score = 1 * probs[0] + 0.5 * probs[1] + 0.0 * probs[2]
        overall_color = np.mean(results[0].orig_img, axis=(0, 1)).astype(np.uint8)
        actual_name, closest_name = get_colour_name(overall_color)
        draw.text((10, 10), f"Water Clarity Score: {overall_score:.2f}, color: {closest_name}", 
                 fill="white", font=fnt, stroke_width=20, stroke_fill="black")
    
    # Store processed image in cache
    app.image_cache[cache_key] = img
    
    return img

# Modify the layout section - move loading screen to be first child of root div
app.layout = html.Div([
    # Loading screen - should be first
    html.Div(
        id='loading-screen',
        children=[
            html.Div(
                children=[
                    html.H3("Processing Images...", style={
                        'marginBottom': '1rem',
                        'color': '#333',
                        'fontSize': '1.5rem'
                    }),
                    html.Div(className="spinner"),
                    html.Div(
                        "This may take a few moments...",
                        style={
                            'marginTop': '1rem',
                            'color': '#666'
                        }
                    )
                ],
                className="loading-content"
            )
        ],
        style={
            'display': 'none',
            'opacity': '0',
            'visibility': 'hidden'
        }
    ),
    
    # Title
    html.Div(
        children=[
            html.Div([
                html.H1("Eyes on the water TV", style={'margin': '0'}),
                html.Img(src='assets/Cover_Logos.png', style={'height': '50px'}),
            ], className='title-section', style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'width': '100%',
                'padding': '5px',
                'backgroundColor': '#f8f9fa',
            }),
        ], className='header-title'
    ),

    # Main content area
    html.Div([
        # Image and Info container
        html.Div([
            # Left section - Analysis plot and Image display
            html.Div([
                # Analysis plot
                dcc.Graph(
                    id='analysis-plot',
                    config={'displayModeBar': True},
                    style={
                        'height': '30vh',
                        'width': '100%',
                        'marginBottom': '20px'
                    }
                ),
                # Image display
                html.Img(
                    id='displayed-image',
                    style={
                        'width': '100%',
                        'maxHeight': '60vh',
                        'objectFit': 'contain'
                    }
                ),
            ], style={
                'width': '80%',
                'display': 'inline-block',
                'verticalAlign': 'top'
            }),
            
            # Right section - Info, Navigation, and settings
            html.Div([
                # Image info section
                html.Div([
                    html.H4('Image Information'),
                    html.Div(id='image-info'),
                ]),
                
                # Folder selection section
                html.Div([
                    html.H4('Folder Selection'),
                    # Parent folder selection
                    html.Div([
                        html.Label('Parent Folder:'),
                        dcc.Input(
                            id='parent-folder-input',
                            type='text',
                            value=image_parent_folder,
                            placeholder='Enter parent folder path',
                            style={'width': '100%', 'marginBottom': '10px'}
                        ),
                    ]),
                    
                    # Level 1 folder dropdown
                    html.Div([
                        html.Label('Main Folder:'),
                        dcc.Dropdown(
                            id='folder-dropdown-l1',
                            options=[],
                            placeholder='Select main folder',
                            style={'width': '100%', 'marginBottom': '10px'}
                        ),
                    ]),
                    
                    # Level 2 folder dropdown
                    html.Div([
                        html.Label('Sub Folder:'),
                        dcc.Dropdown(
                            id='folder-dropdown-l2',
                            options=[],
                            placeholder='Select subfolder',
                            style={'width': '100%', 'marginBottom': '10px'}
                        ),
                    ]),
                ], style={'marginBottom': '20px'}),
                
                # Navigation section
                html.Div([
                    html.H4('Navigation'),
                    # Navigation buttons in a vertical stack
                    html.Button('Previous', 
                        id='prev-button',
                        className='nav-button',
                        n_clicks=0),
                    html.Button('Next', 
                        id='next-button',
                        className='nav-button',
                        n_clicks=0),
                        
                    html.Button('Object detection', 
                        id={'type': 'toggle-button', 'index': 0},
                        className='nav-button toggle-button',
                        n_clicks=0),
                    html.Button('Segmentation', 
                        id={'type': 'toggle-button', 'index': 1},
                        className='nav-button toggle-button',
                        n_clicks=0),
                    html.Button('Water Clarity Index', 
                        id={'type': 'toggle-button', 'index': 2},
                        className='nav-button toggle-button',
                        n_clicks=0),
                ], style={'marginBottom': '20px'}),
                
                # Settings section
                html.Div([
                    html.H4('Settings'),
                    html.Div(id='settings-container', style={'display': 'none'})
                ]),
                
                # Action buttons section
                html.Div([
                    html.Div([
                        html.Button(
                            'Set Current as Baseline', 
                            id='set-baseline-button',
                            disabled=True,
                            style={
                                'width': '100%',
                                'marginBottom': '10px',
                                'padding': '5px',
                                'backgroundColor': '#28a745',
                                'color': 'white',
                                'border': 'none',
                                "white-space": "pre",
                                'borderRadius': '0.25rem',
                                'cursor': 'pointer',
                            },
                        ),
                        dbc.Tooltip(
                            "Please toggle segmentation model first",
                            target="set-baseline-button",
                            placement="top",
                            style={
                                "backgroundColor": "#6c757d",
                                "color": "white",
                                "fontSize": "0.9rem",
                                "padding": "8px",
                            }
                        ),
                    ], style={'width': '100%'}),
                    html.Button(
                        'Save Results', 
                        id='save-results-button',
                        disabled=True,
                        style={
                            'width': '100%',  # Full width
                            'marginBottom': '10px',
                            'padding': '8px',  # Match nav-button padding
                            'backgroundColor': '#6c757d',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',  # Match nav-button border-radius
                            'cursor': 'not-allowed',
                            'textAlign': 'center',  # Center text
                        }
                    ),
                    dcc.Download(id='download-results')
                ], style={'marginTop': '20px'}),
            ], style={
                'width': '18%',
                'display': 'inline-block',
                'marginLeft': '2%',
                'verticalAlign': 'top'
            })
        ], style={
            'width': '100%',
            'display': 'flex',
            'alignItems': 'start'
        }),
    ]),
    
    # Add store for click data
    dcc.Store(id='click-data', data=None),
    
    # Add settings store
    dcc.Store(id='settings-store', data=DEFAULT_SETTINGS),
    
    # Store components for maintaining state
    dcc.Store(id='current-images'),
    dcc.Store(id='current-index', data=0),
    dcc.Store(id='toggle-state', data={'active': None}),
    # Store for detection results
    dcc.Store(id='detection-results', data=None),
    dcc.Store(id='water-quality-results', data=None),
    dcc.Store(id='processed-results-cache', data={
        'folder_path': None,
        'active_toggle': None,
        'settings': None,
        'results': {}  # Will store {image_name: {'detections': n} or {'score': s, 'color': c}}
    }),
    dcc.Store(id='dummy-output'),  # For cache clearing callback
    # Add a store for baselines
    dcc.Store(id='baselines-store', data={}),
])

# region Initialize the AI models
models = model_init()
# endregion

# region Callback to update folder options
@app.callback(
    Output('folder-dropdown-l1', 'options', allow_duplicate=True),
    Output('folder-dropdown-l1', 'value', allow_duplicate=True),
    Output({'type': 'toggle-button', 'index': ALL}, 'style', allow_duplicate=True),
    Output('toggle-state', 'data', allow_duplicate=True),
    Input('parent-folder-input', 'value'),
    prevent_initial_call="initial_duplicate"
)
def update_folders_from_parent(parent_path):

    base_style = {
        'margin': '0.5rem',
        'padding': '0.5rem 1rem',
        'minWidth': '8rem',
        'flex': '1',
    }

    if not parent_path or not os.path.exists(parent_path):
        return [], None, [base_style for _ in range(3)], {'active': None}
    

    if parent_path[-1] != '/':
        parent_path = parent_path + '/'

    # Update global variable
    global image_parent_folder
    image_parent_folder = parent_path
    
    # Get folders from new parent path
    folders = get_subfolders(parent_path)

    return (
        [{'label': folder, 'value': folder} for folder in folders],
        None,
        [base_style for _ in range(3)],
        {'active': None}
    )

@app.callback(
    Output('folder-dropdown-l2', 'options'),
    Output('folder-dropdown-l2', 'value'),
    Input('folder-dropdown-l1', 'value'),
    State('folder-dropdown-l2', 'value'),
    prevent_initial_call=True
)
def update_l2_folder_options(selected_l1_folder, selected_l2_folder):
    if not selected_l1_folder:
        return [], None
    
    parent_folder = os.path.join(image_parent_folder, selected_l1_folder)
    folders = get_subfolders(parent_folder)
    
    return [{'label': folder, 'value': folder} for folder in folders], folders[0]

# endregion

# region Callback to update image list when folder is selected
@app.callback(
    Output('current-images', 'data'),
    Output('current-index', 'data', allow_duplicate=True),
    Output('loading-screen', 'style', allow_duplicate=True),
    Input('folder-dropdown-l2', 'value'),
    State('folder-dropdown-l1', 'value'),
    prevent_initial_call=True
)
def update_image_list(selected_l2_folder, selected_l1_folder):
    if not selected_l2_folder or not selected_l1_folder:
        return [], 0, {'display': 'none'}
    
    folder_path = os.path.join(image_parent_folder, 
                              selected_l1_folder, selected_l2_folder)
    images = get_images(folder_path)
    
    # Show loading screen
    return images, 0, {
        'display': 'flex',
        'opacity': '1',
        'visibility': 'visible'
    }

# endregion

# region Callback to update displayed image and info
@app.callback(
    Output('displayed-image', 'src'),
    Output('analysis-plot', 'figure'),
    Output('image-info', 'children'),
    Output('current-index', 'data', allow_duplicate=True),
    Output('loading-screen', 'style', allow_duplicate=True),
    Input('current-images', 'data'),
    Input('current-index', 'data'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    Input('toggle-state', 'data'),
    Input('settings-store', 'data'),
    Input('processed-results-cache', 'data'),
    Input('analysis-plot', 'clickData'),
    State('folder-dropdown-l1', 'value'),
    State('folder-dropdown-l2', 'value'),
    prevent_initial_call=True
)
def update_image(images, current_index, prev_clicks, next_clicks, 
                toggle_state, settings_store, results_cache,
                click_data, selected_l1_folder, selected_l2_folder):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not images or not selected_l1_folder or not selected_l2_folder:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[{
                'text': 'Please select the folders',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False
            }]
        )
        
        info = html.Div([
            html.P("Filename: None"),
            html.P("Size: N/A"),
            html.P("Format: N/A"),
            html.P("Please select the folders"),
        ])
        
        return '', empty_fig, info, 0, {'display': 'none'}

    # Extract active_toggle from toggle_state
    active_toggle = toggle_state['active'] if toggle_state else None

    # Handle navigation and click events
    if trigger_id == 'prev-button' and current_index > 0:
        current_index -= 1
    elif trigger_id == 'next-button' and current_index < len(images) - 1:
        current_index += 1
    elif trigger_id == 'analysis-plot' and click_data:
        clicked_index = click_data['points'][0]['pointNumber']
        if 0 <= clicked_index < len(images):
            current_index = clicked_index

    # Create analysis figure using the helper function
    baseline = get_baseline(selected_l1_folder)
    if baseline is None and images:
        first_image = images[0]
        baseline = results_cache['results'][first_image]['segmentation']['area']
    
    analysis_fig = create_analysis_plot(
        results_cache, images, current_index, toggle_state, baseline
    )

    # Get current image path and basic info
    image_path = os.path.join(image_parent_folder, selected_l1_folder, selected_l2_folder, images[current_index])
    img = Image.open(image_path)
    
    # Process image (no caching for displayed images)
    processed_img = process_image(img.copy(), active_toggle, models, settings_store, image_path)
    
    # Convert to base64
    buffered = BytesIO()
    processed_img.save(buffered, format=img.format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_src = f'data:image/{img.format.lower()};base64,{img_str}'

    # Update info display
    info = html.Div([
        html.P(f"Filename: {images[current_index]}"),
        html.P(f"Size: {img.size[0]} x {img.size[1]} pixels"),
        html.P(f"Format: {img.format}"),
        html.P(f"Image {current_index + 1} of {len(images)}"),
        html.P(f"Active Toggle: {active_toggle if active_toggle else 'None'}"),
    ])

    return img_src, analysis_fig, info, current_index, {
        'display': 'none',
        'opacity': '0',
        'visibility': 'hidden'
    }

# endregion

# region Callback for toggle buttons
@app.callback(
    Output({'type': 'toggle-button', 'index': ALL}, 'className'),
    Output('toggle-state', 'data'),
    Output('loading-screen', 'style', allow_duplicate=True),
    [Input({'type': 'toggle-button', 'index': ALL}, 'n_clicks')],
    State('toggle-state', 'data'),
    prevent_initial_call=True
)
def update_toggles(n_clicks_list, current_state):
    ctx = dash.callback_context
    
    base_class = 'nav-button toggle-button'
    active_class = 'nav-button toggle-button active'

    if not ctx.triggered:
        return [base_class for _ in range(3)], {'active': None}, {'display': 'none'}

    if not ctx.triggered_id:
        return [base_class for _ in range(3)], {'active': None}, {'display': 'none'}

    clicked_id = ctx.triggered_id['index']
    button_names = ['Object detection', 'Segmentation', 'Water Clarity Index']
    button_id = button_names[clicked_id]
    
    if current_state['active'] == button_id:
        return [base_class for _ in range(3)], {'active': None}, {'display': 'none'}

    classes = []
    for i in range(3):
        if i == clicked_id:
            classes.append(active_class)
        else:
            classes.append(base_class)
    
    # Show loading screen when toggling a model
    return classes, {'active': button_id}, {
        'display': 'flex',
        'opacity': '1',
        'visibility': 'visible'
    }

# endregion

# region Add callback to update settings container
@app.callback(
    Output('settings-container', 'children'),
    Output('settings-container', 'style'),
    Input('toggle-state', 'data'),
    Input('settings-store', 'data')
)
def update_settings_container(toggle_state, current_settings):
    if not toggle_state['active']:
        return [], {'display': 'none'}
    
    active_toggle = toggle_state['active']
    settings = current_settings[active_toggle]
    
    if active_toggle == 'Object detection':
        return html.Div([
            html.Div([
                html.Label(f'Confidence Threshold: {settings["confidence"]:.2f}'),
                dcc.Slider(
                    id='confidence-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    value=settings['confidence'],
                    marks={i/10: str(i/10) for i in range(0, 11, 2)},
                ),
            ], style={'marginBottom': '1rem'}),
            
            html.Div([
                dcc.Checklist(
                    id='od-display-options',
                    options=[
                        {'label': 'Show Overlay', 'value': 'show_overlay'},
                    ],
                    value=['show_overlay'] if settings['show_overlay'] else [],
                )
            ])
        ]), {'display': 'block'}

    elif active_toggle == 'Segmentation':
        return html.Div([
            html.Div([
                html.Label(f'Point X Position: {settings["point_x"]:.2f}'),
                dcc.Slider(
                    id='point-x-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    value=settings['point_x'],
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                ),
            ], style={'marginBottom': '1rem'}),
            
            html.Div([
                html.Label(f'Point Y Position: {settings["point_y"]:.2f}'),
                dcc.Slider(
                    id='point-y-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    value=settings['point_y'],
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                ),
            ], style={'marginBottom': '1rem'}),
            
            html.Div([
                dcc.Checklist(
                    id='seg-display-options',
                    options=[
                        {'label': 'Show Overlay', 'value': 'show_overlay'},
                    ],
                    value=['show_overlay'] if settings['show_overlay'] else [],
                )
            ])
        ]), {'display': 'block'}

    elif active_toggle == 'Water Clarity Index':
        return html.Div([
            html.Div([
                html.Label('Quality Thresholds'),
            ], style={'marginBottom': '1rem'}),
            
            html.Div([
                dcc.Checklist(
                    id='wci-display-options',
                    options=[
                        {'label': 'Show Score', 'value': 'show_score'},
                    ],
                    value=['show_score'] if settings['show_score'] else [],
                )
            ])
        ]), {'display': 'block'}
    
    return [], {'display': 'none'}
# endregion

# region Callback for Object Detection settings
@app.callback(
    Output('settings-store', 'data', allow_duplicate=True),
    [
     Input('confidence-slider', 'value'),
     Input('od-display-options', 'value')
     ],
    State('toggle-state', 'data'),
    State('settings-store', 'data'),
    prevent_initial_call=True
)
def update_object_detection_settings(conf, od_options, toggle_state, current_settings):
    if not toggle_state['active'] or toggle_state['active'] != 'Object detection':
        return dash.no_update

    current_settings['Object detection'].update({
        'confidence': conf if conf is not None else current_settings['Object detection']['confidence'],
        'show_overlay': 'show_overlay' in (od_options or [])
    })
    return current_settings

# endregion

# region Callback for Segmentation settings
@app.callback(
    Output('settings-store', 'data', allow_duplicate=True),
    [
        Input('point-x-slider', 'value'),
        Input('point-y-slider', 'value'),
        Input('seg-display-options', 'value')
        ],
    State('toggle-state', 'data'),
    State('settings-store', 'data'),
    prevent_initial_call=True
)
def update_segmentation_settings(point_x, point_y, seg_options, toggle_state, current_settings):
    if not toggle_state['active'] or toggle_state['active'] != 'Segmentation':
        return dash.no_update
    
    current_settings['Segmentation'].update({
        'point_x': point_x if point_x is not None else current_settings['Segmentation']['point_x'],
        'point_y': point_y if point_y is not None else current_settings['Segmentation']['point_y'],
        'show_overlay': 'show_overlay' in (seg_options or [])
    })
    return current_settings

# endregion

# region Callback for Water Clarity Index settings
@app.callback(
    Output('settings-store', 'data', allow_duplicate=True),
    [
        Input('wci-display-options', 'value')
        ],
    State('toggle-state', 'data'),
    State('settings-store', 'data'),
    prevent_initial_call=True
)
def update_wci_settings(wci_options, toggle_state, current_settings):
    if not toggle_state['active'] or toggle_state['active'] != 'Index':
        return dash.no_update

    current_settings['Water Clarity Index'].update({
        'show_score': 'show_score' in (wci_options or [])
    })

    return current_settings
# endregion

# region Add new callback to process and cache results
@app.callback(
    Output('processed-results-cache', 'data'),
    Output('loading-screen', 'style', allow_duplicate=True),
    Input('current-images', 'data'),  # Changed from folder-dropdown-l2
    Input('settings-store', 'data'),
    State('folder-dropdown-l1', 'value'),
    State('folder-dropdown-l2', 'value'),  # Changed to State
    prevent_initial_call=True
)
def cache_processed_results(images, settings_store, selected_l1_folder, selected_l2_folder):
    if not images or not selected_l2_folder or not selected_l1_folder:
        return {
            'folder_path': None,
            'settings': None,
            'results': {}
        }, {
            'display': 'none',
            'opacity': '0',
            'visibility': 'hidden'
        }
    
    folder_path = os.path.join(image_parent_folder, selected_l1_folder, selected_l2_folder)
    
    # Standardize cache path construction
    cache_folder = os.path.join(selected_l1_folder, selected_l2_folder)
    point_x = settings_store['Segmentation']['point_x']
    point_y = settings_store['Segmentation']['point_y']
    seg_cache_folder = os.path.join(
        "cache", "segmentation",
        cache_folder,
        f"point_{point_x:.2f}_{point_y:.2f}"
    )
    os.makedirs(seg_cache_folder, exist_ok=True)

    current_state = {
        'folder_path': folder_path,
        'settings': settings_store,
        'results': {}
    }
    
    # Process all images with all models
    for img_name in tqdm(images):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)
        current_state['results'][img_name] = {}
        
        # Object Detection
        od_result = models['object_detection'].predict(
            img,
            conf=settings_store['Object detection']['confidence'],
            verbose=False
        )
        detected_classes = [models['object_detection'].names[cls] for cls in od_result[0].boxes.cls.cpu().tolist()]
        detected_confidences = od_result[0].boxes.conf.cpu().tolist()
        current_state['results'][img_name]['object_detection'] = {
            'detections': len(od_result[0].boxes.cls),
            'detected_classes': detected_classes,
            'detected_confidences': detected_confidences
        }
        
        # Segmentation
        mask_path = os.path.join(seg_cache_folder, os.path.basename(img_path).lower().replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'))
        
        if not os.path.exists(mask_path):
            # Calculate point position based on image size and settings
            point_x = int(img.size[0] * settings_store['Segmentation']['point_x'])
            point_y = int(img.size[1] * settings_store['Segmentation']['point_y'])

            # print(f"Segmentation prediction using point: ({point_x}, {point_y})")
            seg_result = models['segmentation'].predict(
                img,
                points=[[point_x, point_y]],
                verbose=False
            )
            idx_with_biggest_area = np.argmax(seg_result[0].masks.data.cpu().numpy().sum(axis=(1, 2)))
            np.save(mask_path, seg_result[0].masks.data.cpu().numpy()[idx_with_biggest_area])
        
        single_mask = np.load(mask_path)
        current_state['results'][img_name]['segmentation'] = {
            'area': float(single_mask.sum())  # Convert to float for JSON serialization
        }

        # Water Clarity Index
        wci_result = models['water_clearity_index'].predict(img, verbose=False)
        probs = wci_result[0].probs.data.cpu().numpy()
        overall_score = 1 * probs[0] + 0.5 * probs[1] + 0.0 * probs[2]
        overall_color = np.mean(wci_result[0].orig_img, axis=(0, 1)).astype(np.uint8)
        _, closest_name = get_colour_name(overall_color)
        
        current_state['results'][img_name]['water_clarity_index'] = {
            'score': float(overall_score),
            'color': closest_name
        }

    # Return results and hide loading screen
    return current_state, {
        'display': 'none',
        'opacity': '0',
        'visibility': 'hidden'
    }

# endregion

# Add a callback to clear the cache when needed
@app.callback(
    Output('dummy-output', 'data'),  # Add this store to your layout
    Input('folder-dropdown-l1', 'value'),
    Input('folder-dropdown-l2', 'value'),
    Input('settings-store', 'data'),
    prevent_initial_call=True
)
def clear_image_cache(folder_l1, folder_l2, settings):
    app.image_cache = {}
    return None

# Add this callback after the other callbacks:
@app.callback(
    Output('download-results', 'data'),
    Input('save-results-button', 'n_clicks'),
    State('processed-results-cache', 'data'),
    State('folder-dropdown-l1', 'value'),
    State('folder-dropdown-l2', 'value'),
    State('toggle-state', 'data'),
    prevent_initial_call=True
)
def download_results(n_clicks, results_cache, folder_l1, folder_l2, toggle_state):
    if not n_clicks or not results_cache or not results_cache['results']:
        return None
    
    active_toggle = toggle_state['active']
    if not active_toggle:
        return None
    
    # Create DataFrame based on the active toggle
    if active_toggle == 'Object detection':
        df = pd.DataFrame([
            {
                'Image': img_name,
                'Number_of_Detections': data['object_detection']['detections'],
                'Detected_Classes': ', '.join(data['object_detection']['detected_classes']),
                'Confidences': ', '.join([f'{conf:.2f}' for conf in data['object_detection']['detected_confidences']])
            }
            for img_name, data in results_cache['results'].items()
        ])
    
    elif active_toggle == 'Segmentation':
        baseline = get_baseline(folder_l1)
        if baseline is None and results_cache['results']:
            first_image = next(iter(results_cache['results']))
            baseline = results_cache['results'][first_image]['segmentation']['area']
            
        df = pd.DataFrame([
            {
                'Image': img_name,
                'Segmented_Area': data['segmentation']['area'],
                'Baseline': baseline,
                'Difference_from_Baseline': data['segmentation']['area'] - baseline
            }
            for img_name, data in results_cache['results'].items()
        ])
    
    elif active_toggle == 'Water Clarity Index':
        df = pd.DataFrame([
            {
                'Image': img_name,
                'Water_Clarity_Score': data['water_clarity_index']['score'],
                'Dominant_Color': data['water_clarity_index']['color']
            }
            for img_name, data in results_cache['results'].items()
        ])
    
    # Generate filename
    filename = f"{folder_l1}_{folder_l2}_{active_toggle.replace(' ', '_')}_results.csv"
    
    return dcc.send_data_frame(df.to_csv, filename, index=False, float_format='%.2f')

# Add this new callback after the other callbacks:
@app.callback(
    Output('save-results-button', 'disabled'),
    Output('save-results-button', 'style'),
    Input('toggle-state', 'data')
)
def update_save_button_state(toggle_state):
    base_style = {
        'margin': '1rem 0',
        'padding': '0.5rem 1rem',
        'color': 'white',
        'border': 'none',
        'borderRadius': '0.25rem',
    }
    
    if not toggle_state or not toggle_state['active']:
        # Disabled state
        return True, {
            **base_style,
            'backgroundColor': '#6c757d',  # Grey
            'cursor': 'not-allowed',
            'opacity': '0.65'
        }
    else:
        # Enabled state
        return False, {
            **base_style,
            'backgroundColor': '#28a745',  # Green
            'cursor': 'pointer'
        }

# Add callback to initialize baselines store
@app.callback(
    Output('baselines-store', 'data'),
    Input('folder-dropdown-l1', 'value'),
    prevent_initial_call=True
)
def initialize_baselines(level1_folder):
    if not level1_folder:
        return {}
    return load_baselines()

# Modify the handle_baseline_setting callback
@app.callback(
    Output('baselines-store', 'data', allow_duplicate=True),
    Output('set-baseline-button', 'disabled'),
    Output('analysis-plot', 'figure', allow_duplicate=True),
    Input('set-baseline-button', 'n_clicks'),
    Input('toggle-state', 'data'),
    State('folder-dropdown-l1', 'value'),
    State('current-images', 'data'),
    State('current-index', 'data'),
    State('processed-results-cache', 'data'),
    State('baselines-store', 'data'),
    prevent_initial_call=True
)
def handle_baseline_setting(n_clicks, toggle_state, level1_folder, images, current_index, 
                          results_cache, current_baselines):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Enable button only when Segmentation is active
    button_disabled = (not toggle_state or 
                      toggle_state['active'] != 'Segmentation' or 
                      not images or 
                      not level1_folder or 
                      not results_cache or 
                      not results_cache['results'])
    
    # Initialize with current plot
    analysis_fig = dash.no_update
    
    if trigger_id == 'set-baseline-button' and n_clicks and not button_disabled and images:
        current_image = images[current_index]
        if current_image in results_cache['results']:
            current_area = results_cache['results'][current_image]['segmentation']['area']
            set_baseline(level1_folder, current_area)
            current_baselines = load_baselines()
            
            # Create updated plot
            analysis_fig = create_analysis_plot(
                results_cache, images, current_index, toggle_state, current_area
            )
    
    return current_baselines, button_disabled, analysis_fig

# Add callback to update set-baseline button state based on toggle
@app.callback(
    Output('set-baseline-button', 'style'),
    Input('toggle-state', 'data')
)
def update_baseline_button_style(toggle_state):
    base_style = {
        'width': '100%',
        'marginBottom': '10px',
        'padding': '5px',
        'color': 'white',
        'border': 'none',
        'borderRadius': '0.25rem',
        'cursor': 'pointer',
    }
    
    if toggle_state and toggle_state['active'] == 'Segmentation':
        return {
            **base_style,
            'backgroundColor': '#28a745',  # Green
            'opacity': '1',
            'cursor': 'pointer',
        }
    else:
        return {
            **base_style,
            'backgroundColor': '#6c757d',  # Grey
            'opacity': '0.65',
            'cursor': 'not-allowed',
        }

# Add a new helper function to create the analysis plot
def create_analysis_plot(results_cache, images, current_index, toggle_state, baseline=None):
    analysis_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if results_cache and results_cache['results']:
        x_values = list(range(len(images)))
        
        # Object Detection trace (primary y-axis)
        y_od = [results_cache['results'][img]['object_detection']['detections'] for img in images]
        analysis_fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_od,
                name='Detections',
                line=dict(color='blue'),
                hovertemplate='Image %{x}<br>Detections: %{y}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Segmentation trace (secondary y-axis)
        y_seg = [results_cache['results'][img]['segmentation']['area'] for img in images]
        analysis_fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_seg,
                name='Segmented Area',
                line=dict(color='red'),
                hovertemplate='Image %{x}<br>Area: %{y:,.0f} pixels<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Water Clarity Index trace (primary y-axis)
        y_wci = [results_cache['results'][img]['water_clarity_index']['score'] for img in images]
        analysis_fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_wci,
                name='Water Clarity',
                line=dict(color='green'),
                hovertemplate='Image %{x}<br>Score: %{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add baseline if provided
        if baseline is not None:
            analysis_fig.add_trace(
                go.Scatter(
                    x=[0, len(images)-1],
                    y=[baseline, baseline],
                    name='Baseline',
                    line=dict(
                        color='black',
                        dash='dash',
                    ),
                    hovertemplate='Baseline: %{y:,.0f} pixels<extra></extra>'
                ),
                secondary_y=True
            )
        
        # Always add all three current image markers
        # Object Detection marker
        analysis_fig.add_trace(
            go.Scatter(
                x=[current_index],
                y=[y_od[current_index]],
                mode='markers',
                marker=dict(color='blue', size=12, symbol='x'),
                name='Current Detection',
                showlegend=False
            ),
            secondary_y=False
        )
        
        # Segmentation marker
        analysis_fig.add_trace(
            go.Scatter(
                x=[current_index],
                y=[y_seg[current_index]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name='Current Area',
                showlegend=False
            ),
            secondary_y=True
        )
        
        # Water Clarity marker
        analysis_fig.add_trace(
            go.Scatter(
                x=[current_index],
                y=[y_wci[current_index]],
                mode='markers',
                marker=dict(color='green', size=12, symbol='x'),
                name='Current Clarity',
                showlegend=False
            ),
            secondary_y=False
        )

        # Update layout
        analysis_fig.update_layout(
            title='Multi-Model Analysis',
            xaxis_title='Image Index',
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            clickmode='event',
            legend=dict(
                orientation="h",  # horizontal orientation
                yanchor="bottom",
                y=-0.2,  # position below the plot
                xanchor="center",
                x=0.5    # centered horizontally
            )
        )
        
        # Update y-axes titles
        analysis_fig.update_yaxes(title_text="Detections / Water Clarity Score", secondary_y=False)
        analysis_fig.update_yaxes(title_text="Segmented Area (pixels)", secondary_y=True)
    
    return analysis_fig

# Add a clientside callback to prevent scrolling when loading
app.clientside_callback(
    """
    function(style) {
        if (style && style.display === 'flex') {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'auto';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-output', 'data', allow_duplicate=True),
    Input('loading-screen', 'style'),
    prevent_initial_call=True
)

if __name__ == '__main__':
    # app.run_server(host="0.0.0.0", port="8051", debug=True)
    
    # Change the port if port 8051 is occupied
    app.run_server(port="8051", debug=True)
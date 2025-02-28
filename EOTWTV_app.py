from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64
import diskcache
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from baseline_config import load_baselines, set_baseline, get_baseline
import json

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
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
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

image_parent_folder = 'sample_data/Camera1/TIMEL0001'

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
    'Water clarity Index': {
        'show_score': True,
    }
}

# Load tooltips from JSON file
with open('assets/tooltips.json', 'r') as f:
    TOOLTIPS = json.load(f)

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
        
    elif active_toggle == 'Water clarity Index':
        results = models['water_clearity_index'].predict(img, verbose=False)
        probs = results[0].probs.data.cpu().numpy()
        overall_score = 1 * probs[0] + 0.5 * probs[1] + 0.0 * probs[2]
        overall_color = np.mean(results[0].orig_img, axis=(0, 1)).astype(np.uint8)
        actual_name, closest_name = get_colour_name(overall_color)
        draw.text((10, 10), f"Water clarity score: {overall_score:.2f}, color: {closest_name}", 
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
                html.H1("Eyes on the Water TV", style={'margin': '0'}),
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
                        'marginBottom': '10px'
                    }
                ),
                # Plot controls
                html.Div([
                    html.Div([
                        html.Label("Select data to display:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Checklist(
                            id='plot-controls',
                            options=[
                                {'label': 'Object Detection', 'value': 'object_detection'},
                                {'label': 'Segmentation Area', 'value': 'segmentation'},
                                {'label': 'Water Clarity Score', 'value': 'water_clarity'},
                                {'label': 'Baseline', 'value': 'baseline'}
                            ],
                            value=['object_detection', 'segmentation', 'water_clarity', 'baseline'],
                            inline=True,
                            style={'display': 'flex', 'gap': '15px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'})
                ]),
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
                
                
                # Folder selection section
                html.Div([
                    html.H4('Folder Selection'),
                    dbc.Button(
                        "Select Folder", 
                        id="folder-select-button",
                        color="primary",
                        className="mr-1",
                        style={
                            'width': '100%',
                            'marginBottom': '10px',
                            'padding': '10px',
                        }
                    ),
                    html.Div(id="selected-folder-display", style={'marginTop': '10px'}),
                ], style={'marginBottom': '20px'}),

                # Image info section
                html.Div([
                    html.H4('Image Information'),
                    html.Div(id='image-info'),
                ]),
                
                # Navigation section
                html.Div([
                    html.H4('Navigation'),
                    # Navigation buttons in a vertical stack
                    html.Button('Previou image', 
                        id='prev-button',
                        className='nav-button',
                        n_clicks=0),
                    html.Button('Next image', 
                        id='next-button',
                        className='nav-button',
                        n_clicks=0),
                    
                    html.H4('Analysis modes'),
                    html.Div([
                        html.I(className="fas fa-info-circle", id="od-info-icon", 
                               style={"marginLeft": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                        html.Button('Object Detection', 
                            id={'type': 'toggle-button', 'index': 0},
                            className='nav-button toggle-button',
                            n_clicks=0),
                        
                    ], style={"display": "flex", "alignItems": "center"}),
                    
                    dbc.Tooltip(
                        TOOLTIPS["analysis_modes"]["object_detection"]["description"],
                        target="od-info-icon",
                        placement="right"
                    ),
                    
                    html.Div([
                        html.I(className="fas fa-info-circle", id="seg-info-icon", 
                               style={"marginLeft": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                        html.Button('Segmentation', 
                            id={'type': 'toggle-button', 'index': 1},
                            className='nav-button toggle-button',
                            n_clicks=0),
                        
                    ], style={"display": "flex", "alignItems": "center"}),
                    
                    dbc.Tooltip(
                        TOOLTIPS["analysis_modes"]["segmentation"]["description"],
                        target="seg-info-icon",
                        placement="right"
                    ),
                    
                    html.Div([
                        html.I(className="fas fa-info-circle", id="wci-info-icon", 
                               style={"marginLeft": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                        html.Button('Water Clarity Index', 
                            id={'type': 'toggle-button', 'index': 2},
                            className='nav-button toggle-button',
                            n_clicks=0),
                        
                    ], style={"display": "flex", "alignItems": "center"}),
                    
                    dbc.Tooltip(
                        TOOLTIPS["analysis_modes"]["water_clarity_index"]["description"],
                        target="wci-info-icon",
                        placement="right"
                    ),
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
                            'Set current image as baseline', 
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
                    
                    html.Div([
                        html.Button(
                            'Set average area as baseline', 
                            id='set-avg-baseline-button',
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
                            target="set-avg-baseline-button",
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
                        'Save results', 
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
    # Add a store for selected folder path
    dcc.Store(id='selected-folder-path', data=image_parent_folder),
    # Add a store for plot controls
    dcc.Store(id='plot-controls-state', data=['object_detection', 'segmentation', 'water_clarity', 'baseline']),
    # Add a modal for folder selection
    dbc.Modal(
        [
            dbc.ModalHeader("Select Folder"),
            dbc.ModalBody([
                html.P("Select a folder to analyze:"),
                dbc.Input(
                    id="folder-input",
                    type="text",
                    placeholder="Enter folder path",
                    value=image_parent_folder,
                ),
                html.Div(id="folder-structure", style={"marginTop": "15px"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="close-folder-modal", className="ml-auto"),
                dbc.Button("Select", id="confirm-folder-selection", color="primary"),
            ]),
        ],
        id="folder-selection-modal",
        size="lg",
    ),
])

# region Initialize the AI models
models = model_init()
# endregion

# Modify the create_analysis_plot function to handle the new baseline lookup
def create_analysis_plot(results_cache, images, current_index, toggle_state, baseline=None,
                        selected_folder_path=None, plot_controls=None):
    # Default to showing all if no controls provided
    if plot_controls is None:
        plot_controls = ['object_detection', 'segmentation', 'water_clarity', 'baseline']
    
    # Create a figure with multiple subplots - one for each active y-axis
    subplot_specs = []
    
    # Determine which y-axes we need
    need_od_axis = 'object_detection' in plot_controls
    need_wci_axis = 'water_clarity' in plot_controls
    need_seg_axis = 'segmentation' in plot_controls
    
    # Create subplot specs based on active axes
    if need_od_axis and need_wci_axis:
        # Both object detection and water clarity use the left y-axis
        subplot_specs = [[{"secondary_y": need_seg_axis}]]
    elif need_seg_axis:
        # Only segmentation axis needed
        subplot_specs = [[{"secondary_y": True}]]
    else:
        # Only primary axis needed
        subplot_specs = [[{"secondary_y": False}]]
    
    analysis_fig = make_subplots(specs=subplot_specs)
    
    if results_cache and results_cache['results'] and images:
        x_values = list(range(len(images)))
        
        # Object Detection trace (primary y-axis)
        if 'object_detection' in plot_controls:
            y_od = [results_cache['results'][img]['object_detection']['detections'] for img in images]
            analysis_fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_od,
                    name='Object detected',
                    line=dict(color='blue'),
                    hovertemplate='Image %{x}<br>Detections: %{y}<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add current image marker for object detection
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
        
        # Segmentation trace (secondary y-axis)
        if 'segmentation' in plot_controls:
            y_seg = [results_cache['results'][img]['segmentation']['area'] for img in images]
            analysis_fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_seg,
                    name='Segmented area',
                    line=dict(color='red'),
                    hovertemplate='Image %{x}<br>Area: %{y:,.0f} pixels<extra></extra>'
                ),
                secondary_y=True
            )
            
            # Add current image marker for segmentation
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
        
        # Water Clarity Index trace (primary y-axis)
        if 'water_clarity' in plot_controls:
            y_wci = [results_cache['results'][img]['water_clarity_index']['score'] for img in images]
            analysis_fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_wci,
                    name='Water clarity score',
                    line=dict(color='green'),
                    hovertemplate='Image %{x}<br>Score: %{y:.2f}<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add current image marker for water clarity
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
        
        # Add baseline if provided and requested
        if baseline is not None and 'baseline' in plot_controls and 'segmentation' in plot_controls:
            analysis_fig.add_trace(
                go.Scatter(
                    x=[0, len(images)-1],
                    y=[baseline, baseline],
                    name='Segmented area baseline',
                    line=dict(
                        color='black',
                        dash='dash',
                    ),
                    hovertemplate='Baseline: %{y:,.0f} pixels<extra></extra>'
                ),
                secondary_y=True
            )

        # Update layout
        analysis_fig.update_layout(
            title='Multi-Model Analysis Results',
            xaxis_title='Image Index',
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            clickmode='event',
            legend=dict(
                orientation="h",  # horizontal orientation
                yanchor="bottom",
                y=-0.5,  # position below the plot
                xanchor="center",
                x=0.5    # centered horizontally
            )
        )
        
        # Update y-axes titles based on what's being displayed
        if need_od_axis or need_wci_axis:
            # Determine the appropriate title for the primary y-axis
            primary_title = []
            if need_od_axis:
                primary_title.append("Detections")
            if need_wci_axis:
                primary_title.append("Water clarity score")
            
            analysis_fig.update_yaxes(
                title_text=" / ".join(primary_title), 
                secondary_y=False,
                showgrid=True
            )
        
        if need_seg_axis:
            analysis_fig.update_yaxes(
                title_text="Segmented area (pixels)", 
                secondary_y=True,
                showgrid=True
            )
    else:
        # Empty plot with message
        analysis_fig.update_layout(
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            annotations=[{
                'text': 'No data available',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False
            }]
        )
    
    return analysis_fig

# region Callback to update image list when folder is selected
@app.callback(
    Output('current-images', 'data'),
    Output('current-index', 'data', allow_duplicate=True),
    Output('loading-screen', 'style', allow_duplicate=True),
    Input('folder-select-button', 'n_clicks'),
    State('selected-folder-path', 'data'),
    prevent_initial_call=True
)
def update_selected_folder(n_clicks, selected_folder_path):
    if not selected_folder_path:
        return [], 0, {'display': 'none'}
    
    images = get_images(selected_folder_path)
    
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
    Input('plot-controls', 'value'),
    State('selected-folder-path', 'data'),
    prevent_initial_call=True
)
def update_image(images, current_index, prev_clicks, next_clicks, 
                toggle_state, settings_store, results_cache,
                click_data, plot_controls, selected_folder_path):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not images or not selected_folder_path:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[{
                'text': 'Please select a folder',
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
            html.P("Please select a folder"),
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

    # Create analysis figure using the helper function with selected folder path
    baseline = get_baseline_folder(selected_folder_path)
    
    analysis_fig = create_analysis_plot(
        results_cache, images, current_index, toggle_state, baseline,
        selected_folder_path, plot_controls
    )

    # Get current image path and basic info
    image_path = os.path.join(selected_folder_path, images[current_index])
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
    button_names = ['Object detection', 'Segmentation', 'Water clarity Index']
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
                html.Div([
                    html.I(className="fas fa-info-circle", id="confidence-info-icon", 
                          style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                    html.Label(f'Confidence Threshold: {settings["confidence"]:.2f}'),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"}),
                
                dbc.Tooltip(
                    TOOLTIPS["settings"]["object_detection"]["confidence"]["description"],
                    target="confidence-info-icon",
                    placement="right"
                ),
                
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
                html.Div([
                    html.I(className="fas fa-info-circle", id="od-overlay-info-icon", 
                          style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                    dcc.Checklist(
                        id='od-display-options',
                        options=[
                            {'label': 'Show Overlay', 'value': 'show_overlay'},
                        ],
                        value=['show_overlay'] if settings['show_overlay'] else [],
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
                
                dbc.Tooltip(
                    TOOLTIPS["settings"]["object_detection"]["show_overlay"]["description"],
                    target="od-overlay-info-icon",
                    placement="right"
                ),
            ])
        ]), {'display': 'block'}

    elif active_toggle == 'Segmentation':
        return html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle", id="point-x-info-icon", 
                          style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                    html.Label(f'Point X Position: {settings["point_x"]:.2f}'),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"}),
                
                dbc.Tooltip(
                    TOOLTIPS["settings"]["segmentation"]["point_x"]["description"],
                    target="point-x-info-icon",
                    placement="right"
                ),
                
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
                html.Div([
                    html.I(className="fas fa-info-circle", id="point-y-info-icon", 
                          style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                    html.Label(f'Point Y Position: {settings["point_y"]:.2f}'),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"}),
                
                dbc.Tooltip(
                    TOOLTIPS["settings"]["segmentation"]["point_y"]["description"],
                    target="point-y-info-icon",
                    placement="right"
                ),
                
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
                html.Div([
                    html.I(className="fas fa-info-circle", id="seg-overlay-info-icon", 
                          style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
                    dcc.Checklist(
                        id='seg-display-options',
                        options=[
                            {'label': 'Show Overlay', 'value': 'show_overlay'},
                        ],
                        value=['show_overlay'] if settings['show_overlay'] else [],
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
                
                dbc.Tooltip(
                    TOOLTIPS["settings"]["segmentation"]["show_overlay"]["description"],
                    target="seg-overlay-info-icon",
                    placement="right"
                ),
            ])
        ]), {'display': 'block'}

    elif active_toggle == 'Water clarity Index':
        return html.Div([
            # html.Div([
            #     html.Div([
            #         html.I(className="fas fa-info-circle", id="wci-threshold-info-icon", 
            #               style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
            #         html.Label('Quality Thresholds'),
            #     ], style={"display": "flex", "alignItems": "center"}),
            # ], style={'marginBottom': '1rem'}),
            
            # html.Div([
            #     html.Div([
            #         html.I(className="fas fa-info-circle", id="wci-score-info-icon", 
            #               style={"marginRight": "8px", "color": "#17a2b8", "cursor": "pointer"}),
            #         dcc.Checklist(
            #             id='wci-display-options',
            #             options=[
            #                 {'label': 'Show Score', 'value': 'show_score'},
            #             ],
            #             value=['show_score'] if settings['show_score'] else [],
            #         ),
            #     ], style={"display": "flex", "alignItems": "center"}),
                
            #     dbc.Tooltip(
            #         TOOLTIPS["settings"]["water_clarity_index"]["show_score"]["description"],
            #         target="wci-score-info-icon",
            #         placement="right"
            #     ),
            # ])
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
    if not toggle_state['active'] or toggle_state['active'] != 'Water clarity Index':
        return dash.no_update

    current_settings['Water clarity Index'].update({
        'show_score': 'show_score' in (wci_options or [])
    })

    return current_settings
# endregion

# region Add new callback to process and cache results
@app.callback(
    Output('processed-results-cache', 'data'),
    Output('loading-screen', 'style', allow_duplicate=True),
    Input('current-images', 'data'),
    Input('settings-store', 'data'),
    State('selected-folder-path', 'data'),
    prevent_initial_call=True
)
def cache_processed_results(images, settings_store, selected_folder_path):
    if not images or not selected_folder_path:
        return {
            'folder_path': None,
            'settings': None,
            'results': {}
        }, {
            'display': 'none',
            'opacity': '0',
            'visibility': 'hidden'
        }
    
    folder_path = selected_folder_path
    
    # Standardize cache path construction
    cache_folder = os.path.basename(folder_path)
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
        actual_name, closest_name = get_colour_name(overall_color)
        
        current_state['results'][img_name]['water_clarity_index'] = {
            'score': float(overall_score),
            'color': closest_name
        }
    
    # Check if baseline exists, if not, set the average area as baseline
    if get_baseline_folder(folder_path) is None and images and current_state['results']:
        # Calculate average area
        avg_area = calculate_average_area(current_state)
        if avg_area is not None:
            # Set average as baseline
            set_baseline_folder(folder_path, avg_area, is_average=True)
            print(f"Set average area ({avg_area:.2f}) as baseline for {folder_path}")

    # Return results and hide loading screen
    return current_state, {
        'display': 'none',
        'opacity': '0',
        'visibility': 'hidden'
    }

# endregion

# Add a clear cache callback
@app.callback(
    Output('dummy-output', 'data'),  # Add this store to your layout
    Input('folder-select-button', 'n_clicks'),
    Input('settings-store', 'data'),
    prevent_initial_call=True
)
def clear_image_cache(folder_button, settings):
    app.image_cache = {}
    return None

# Add this callback after the other callbacks:
@app.callback(
    Output('download-results', 'data'),
    Input('save-results-button', 'n_clicks'),
    State('processed-results-cache', 'data'),
    State('selected-folder-path', 'data'),
    State('toggle-state', 'data'),
    prevent_initial_call=True
)
def download_results(n_clicks, results_cache, selected_folder_path, toggle_state):
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
        baseline = get_baseline_folder(selected_folder_path)
        if baseline is None and results_cache['results']:
            first_image = next(iter(results_cache['results']))
            baseline = results_cache['results'][first_image]['segmentation']['area']
            
        df = pd.DataFrame([
            {
                'Image': img_name,
                'Segmented_Area': data['segmentation']['area'],
                'Baseline': baseline,
                'Difference_from_Baseline': data['segmentation']['area'] - baseline,
                'Percent_Change': ((data['segmentation']['area'] - baseline) / baseline * 100) if baseline else 0
            }
            for img_name, data in results_cache['results'].items()
        ])
    
    elif active_toggle == 'Water clarity Index':
        df = pd.DataFrame([
            {
                'Image': img_name,
                'Water_Clarity_Score': data['water_clarity_index']['score'],
                'Dominant_Color': data['water_clarity_index']['color']
            }
            for img_name, data in results_cache['results'].items()
        ])
    
    # Generate filename
    folder_name = os.path.basename(selected_folder_path)
    filename = f"{folder_name}_{active_toggle.replace(' ', '_')}_results.csv"
    
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

# Modify the initialize_baselines callback
@app.callback(
    Output('baselines-store', 'data'),
    Input('selected-folder-path', 'data'),
    prevent_initial_call=True
)
def initialize_baselines(selected_folder_path):
    if not selected_folder_path:
        return {}
    
    # Just load all baselines
    return load_baselines()

# Modify the handle_baseline_setting callback
@app.callback(
    Output('baselines-store', 'data', allow_duplicate=True),
    Output('set-baseline-button', 'disabled'),
    Output('set-avg-baseline-button', 'disabled'),
    Output('set-baseline-button', 'style'),
    Output('set-avg-baseline-button', 'style'),
    Output('analysis-plot', 'figure', allow_duplicate=True),
    Input('set-baseline-button', 'n_clicks'),
    Input('set-avg-baseline-button', 'n_clicks'),
    Input('toggle-state', 'data'),
    State('selected-folder-path', 'data'),
    State('current-images', 'data'),
    State('current-index', 'data'),
    State('processed-results-cache', 'data'),
    State('baselines-store', 'data'),
    State('plot-controls', 'value'),
    prevent_initial_call=True
)
def handle_baseline_setting(n_clicks, avg_n_clicks, toggle_state, selected_folder_path, images, 
                          current_index, results_cache, current_baselines, plot_controls):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Base styles for buttons
    base_style = {
        'width': '100%',
        'marginBottom': '10px',
        'padding': '5px',
        'color': 'white',
        'border': 'none',
        'borderRadius': '0.25rem',
        'whiteSpace': 'pre',
    }
    
    active_style = {
        **base_style,
        'backgroundColor': '#28a745',  # Green
        'opacity': '1',
        'cursor': 'pointer',
    }
    
    inactive_style = {
        **base_style,
        'backgroundColor': '#6c757d',  # Grey
        'opacity': '0.65',
        'cursor': 'not-allowed',
    }
    
    available_style = {
        **base_style,
        'backgroundColor': '#17a2b8',  # Blue
        'opacity': '1',
        'cursor': 'pointer',
    }
    
    # Check if segmentation is active
    segmentation_active = toggle_state and toggle_state['active'] == 'Segmentation'
    
    # Initialize with current plot
    analysis_fig = dash.no_update
    
    # Process button clicks first to ensure baseline type is updated
    if trigger_id == 'set-baseline-button' and n_clicks and segmentation_active and images:
        # Set current image as baseline
        current_image = images[current_index]
        if current_image in results_cache['results']:
            current_area = results_cache['results'][current_image]['segmentation']['area']
            set_baseline_folder(selected_folder_path, current_area, is_average=False)
            
            # Create updated plot
            analysis_fig = create_analysis_plot(
                results_cache, images, current_index, toggle_state, current_area,
                selected_folder_path, plot_controls
            )
    
    elif trigger_id == 'set-avg-baseline-button' and avg_n_clicks and segmentation_active:
        # Set average area as baseline
        avg_area = calculate_average_area(results_cache)
        if avg_area is not None:
            set_baseline_folder(selected_folder_path, avg_area, is_average=True)
            
            # Create updated plot
            analysis_fig = create_analysis_plot(
                results_cache, images, current_index, toggle_state, avg_area,
                selected_folder_path, plot_controls
            )
    
    # Get current baseline type AFTER any changes from button clicks
    baseline_type = get_baseline_type(selected_folder_path) if selected_folder_path else None
    
    # Set button styles based on baseline type
    if not segmentation_active:
        # Both buttons disabled if segmentation is not active
        single_button_style = inactive_style
        avg_button_style = inactive_style
        single_button_disabled = True
        avg_button_disabled = True
    else:
        # Set styles based on current baseline type
        if baseline_type == 'single':
            single_button_style = active_style
            avg_button_style = available_style
            single_button_disabled = False
            avg_button_disabled = False
        elif baseline_type == 'average':
            single_button_style = available_style
            avg_button_style = active_style
            single_button_disabled = False
            avg_button_disabled = False
        else:
            # No baseline set yet
            single_button_style = available_style
            avg_button_style = available_style
            single_button_disabled = False
            avg_button_disabled = False
    
    return current_baselines, single_button_disabled, avg_button_disabled, single_button_style, avg_button_style, analysis_fig

# Add callback to update the tooltips for baseline buttons
@app.callback(
    Output('set-baseline-button', 'children'),
    Output('set-avg-baseline-button', 'children'),
    Input('selected-folder-path', 'data'),
    Input('toggle-state', 'data'),
    Input('set-baseline-button', 'n_clicks'),
    Input('set-avg-baseline-button', 'n_clicks'),
)
def update_baseline_button_labels(selected_folder_path, toggle_state, single_clicks, avg_clicks):
    if not selected_folder_path:
        return 'Set current image as baseline', 'Set average area as baseline'
    
    # Get the current baseline type
    baseline_type = get_baseline_type(selected_folder_path)
    
    if baseline_type == 'single':
        return 'Set current image as baseline', 'Set average area as baseline'
        # return 'Set current image as baseline (Active)', 'Set average area as baseline'
    elif baseline_type == 'average':
        # return 'Set current image as baseline', 'Set average area as baseline (Active)'
        return 'Set current image as baseline', 'Set average area as baseline'
    else:
        return 'Set current image as baseline', 'Set average area as baseline'

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

# Replace the folder selection callbacks with new ones
@app.callback(
    Output("folder-selection-modal", "is_open"),
    [Input("folder-select-button", "n_clicks"), 
     Input("close-folder-modal", "n_clicks"), 
     Input("confirm-folder-selection", "n_clicks")],
    [State("folder-selection-modal", "is_open")],
)
def toggle_folder_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open

@app.callback(
    Output("folder-structure", "children"),
    Input("folder-input", "value"),
)
def update_folder_structure(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        return html.Div("Folder not found", style={"color": "red"})
    
    try:
        # Get all subdirectories and files
        items = os.listdir(folder_path)
        dirs = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(folder_path, item)) 
                and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        return html.Div([
            html.P(f"Found {len(dirs)} folders and {len(files)} images"),
            html.Div([
                html.Strong("Folders:"),
                html.Ul([html.Li(d) for d in dirs[:5]] + 
                       ([html.Li(f"... and {len(dirs) - 5} more")] if len(dirs) > 5 else [])),
            ]) if dirs else html.Div("No subfolders found"),
            html.Div([
                html.Strong("Images:"),
                html.Ul([html.Li(f) for f in files[:5]] + 
                       ([html.Li(f"... and {len(files) - 5} more")] if len(files) > 5 else [])),
            ]) if files else html.Div("No images found"),
        ])
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={"color": "red"})

@app.callback(
    Output("selected-folder-path", "data"),
    Output('current-images', 'data', allow_duplicate=True),
    Output('current-index', 'data', allow_duplicate=True),
    Output('loading-screen', 'style', allow_duplicate=True),
    Input("confirm-folder-selection", "n_clicks"),
    State("folder-input", "value"),
    prevent_initial_call=True
)
def update_selected_folder(n_clicks, folder_path):
    if not n_clicks or not folder_path or not os.path.exists(folder_path):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Update global variable
    global image_parent_folder
    image_parent_folder = folder_path
    
    # Get images from the selected folder
    images = get_images(folder_path)
    
    # Show loading screen
    return folder_path, images, 0, {
        'display': 'flex',
        'opacity': '1',
        'visibility': 'visible'
    }

# Add a callback to update the selected folder display
@app.callback(
    Output("selected-folder-display", "children"),
    Input("selected-folder-path", "data"),
)
def update_selected_folder_display(selected_folder_path):
    if not selected_folder_path:
        return html.Div("No folder selected")
    
    # Count images in the folder
    try:
        image_count = len(get_images(selected_folder_path))
        return html.Div([
            html.P(f"Selected folder: {selected_folder_path}", style={'wordBreak': 'break-word'}),
            html.P(f"Found {image_count} images")
        ])
    except Exception as e:
        return html.Div([
            html.P(f"Selected folder: {selected_folder_path}", style={'wordBreak': 'break-word'}),
            html.P(f"Error: {str(e)}", style={"color": "red"})
        ])

# Define functions to handle baselines with the new folder structure
def get_baseline_folder(folder_path):
    """Get the baseline for a specific folder"""
    # Create folder-specific baseline file path
    baseline_file = os.path.join(folder_path, 'baseline.json')
    
    # If baseline file exists, load it
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                return baseline_data.get('baseline')
        except Exception as e:
            print(f"Error loading baseline file: {e}")
            return None
    
    return None

def get_baseline_type(folder_path):
    """Get the type of baseline (single image or average)"""
    baseline_file = os.path.join(folder_path, 'baseline.json')
    
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                return baseline_data.get('type', 'single')  # Default to 'single' if not specified
        except Exception as e:
            print(f"Error loading baseline file: {e}")
            return 'single'
    
    return None

def set_baseline_folder(folder_path, value, is_average=False):
    """Set the baseline for a specific folder"""
    # Create folder-specific baseline file path
    baseline_file = os.path.join(folder_path, 'baseline.json')
    
    # Create or update the baseline file
    baseline_data = {
        'baseline': value,
        'type': 'average' if is_average else 'single'
    }
    
    try:
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f)
        print(f"Baseline saved to {baseline_file}")
        return True
    except Exception as e:
        print(f"Error saving baseline file: {e}")
        return False

def calculate_average_area(results_cache):
    """Calculate the average segmented area from all images in the folder"""
    if not results_cache or not results_cache['results']:
        return None
    
    areas = [data['segmentation']['area'] for _, data in results_cache['results'].items()]
    if not areas:
        return None
    
    return sum(areas) / len(areas)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=False, type=int, default=8051)
    args = parser.parse_args()
    
    app.run_server(host="0.0.0.0", port=args.port, debug=True)
    # app.run_server(port=args.port, debug=True)
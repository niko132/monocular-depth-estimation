'''
MiDaS code and pretrained model from
https://github.com/isl-org/MiDaS
'''

'''
This script was created by
Niko KIrste
'''

import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colorbar
from datetime import datetime

import open3d as o3d


# prepare the neural network
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# the inference function
def midas_helper(img):
  input_batch = transform(img).to(device)

  with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

  output = prediction.cpu().numpy()
  
  return output


# prepare the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


# all window sizes are based on these dims
width = 640
height = 480


# create point cloud data
x = np.tile(np.arange(width), height)
y = np.repeat(np.arange(height), width)[::-1]

xyz = np.zeros((width * height, 3), dtype=int)
xyz[:,0] = x
xyz[:,1] = y

# prepare the point cloud viewer
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Point Cloud', width=640, height=480)
vis.get_render_option().show_coordinate_frame = True
vis.get_render_option().background_color = np.array([0.9,0.9,0.9])
ctr = vis.get_view_control()
parameters = o3d.io.read_pinhole_camera_parameters("pc_camera_params.json")
ctr.convert_from_pinhole_camera_parameters(parameters)


# create the color bar
fig = plt.figure(figsize=(2,4), dpi=120)
ax = fig.add_axes([0.1, 0.05, 0.2, 0.9])
cb = colorbar.ColorbarBase(ax, orientation='vertical', cmap='plasma')
cb.set_ticks([0.0, 0.5, 1.0])
cb.ax.set_yticklabels(["0.0 - far", "0.5", "1.0 - near"], fontsize=14)
fig.canvas.draw()
# now we can save it to a numpy array.
legend = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
legend = legend.reshape(fig.canvas.get_width_height()[::-1] + (3,))
legend = legend[:,:,[2,1,0]]


# main loop
calc = True
while True:
    t1 = datetime.now()
    
    # create depth map or view it
    if calc:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width,height))
        
        output = midas_helper(frame)
        output = ((output - output.min()) * (1 / (output.max() - output.min()) * 255)).astype(np.uint8)
        
        disparity = output
        
        output = np.uint8(cm.plasma(output) * 255)
        output = output[:,:,[2,1,0]]        
        combined = np.concatenate((output, legend), axis=1)
        
        cv2.imshow('Input', frame)
        cv2.imshow('Disparity', combined)
    else:
        vis.poll_events()
        vis.update_renderer()
    
    # listen for keypress to switch mode
    c = cv2.waitKey(1)
    if c == ord('2'):
        if calc:
            print('Viewer Mode...')
        
            # get the point cloud in the viewer
            xyz[:,2] = disparity.flatten() * 2.0

            rgb = frame.reshape(-1, frame.shape[-1])
            rgb = rgb[:,[2,1,0]]
            rgb = rgb.astype('float') / 255.0
            
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            
            parameters = ctr.convert_to_pinhole_camera_parameters()
            vis.remove_geometry(pcd, False)
            vis.add_geometry(pcd, True)
            ctr.convert_from_pinhole_camera_parameters(parameters)
        
        calc = False
    elif c == ord('1'):
        if not calc:
            print('Calc Mode...')
        
        calc = True
    if c == 27:
        break
    
    fps = 1.0 / (datetime.now() - t1).total_seconds()
    print('FPS: %s' % fps)

# destroy all windows and resources
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PROCESSED_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide/processed'
FIG_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide/figures'

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npy')]
if not files:
    print("No processed files found.")
    exit()

target_file = files[0]
data_path = os.path.join(PROCESSED_DIR, target_file)
print(f"Visualizing {target_file}...")

video_array = np.load(data_path)

frames_to_show = video_array[:, :, :50]

fig = plt.figure()
ims = []
for i in range(frames_to_show.shape[2]):
    im = plt.imshow(frames_to_show[:, :, i], cmap='gray', animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
save_path = os.path.join(FIG_DIR, 'debug_brain_video.gif')
ani.save(save_path, writer='pillow')

print(f"Visualization saved to {save_path}")
print("Please download this GIF and check if you can see brain structures clearly.")
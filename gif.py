import imageio
import os
from pathlib import Path
# You must create a folder named "source_images" and put the pictures inside 
image_path = Path('source_images')
# image_path = Path('source_images2')

images = list(image_path.glob('*.png'))
#sorting files to put in correct order to convert to .gif 
images = sorted(images)
print(images)
image_list = []
# path = os.getcwd()
# for filename in os.listdir(path):
#             if filename.startswith('espacio_fases'):
#                 print(filename)
#                 image_list.append(imageio.imread(filename))
# print(images)
for file_name in images:
    image_list.append(imageio.imread(file_name))

imageio.mimwrite('animated_from_images.mp4', image_list,fps=5)
imageio.mimread('animated_from_images.mp4')

# imageio.mimwrite('pos_from_images.mp4', image_list,fps=5)
# imageio.mimread('pos_from_images.mp4')
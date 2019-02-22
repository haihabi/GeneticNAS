import imageio
import glob
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

images = dict()
epoch_dict = dict()
log_dir = '/data/projects/GNAS/logs/2019_02_17_20_25_42'

font = ImageFont.truetype("/home/haih/Downloads/Untitled Folder/Microsoft Sans Serif.ttf", 16)
for filename in glob.glob(log_dir + "/*.png"):
    layer_index = int(str.split(filename, '_')[-1].split('.png')[0])
    epoch = int(str.split(filename, '_')[-2])
    img = imageio.imread(filename)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    draw.text((0, 0), 'Epoch:' + str(epoch), (0, 0, 0), font=font)
    img = np.array(img_pil.getdata()).reshape(img_pil.size[1], img_pil.size[0], 4)
    if images.get(layer_index) is None:
        images.update({layer_index: [img]})
        epoch_dict.update({layer_index: [epoch]})
    else:
        images.get(layer_index).append(img)
        epoch_dict.get(layer_index).append(epoch)

# Reorder
for k, v in epoch_dict.items():
    index = np.argsort(np.asarray(v)).astype('int')
    images.update({k: [np.asarray(images.get(k))[i] for i in index]})

for k, v in images.items():
    imageio.mimsave(os.path.join(log_dir, 'cell_movie_' + str(k) + '.gif'), v, duration=0.5)

# pathlib.Path path objects are recommended instead of strings
import pathlib

# import pyanime4k library
import pyanime4k
import os

images = []
folder = 'Sailor/EXP3'
for img in os.listdir(folder):
    images.append(folder+'/'+img)

pyanime4k.upscale_images(
    input_paths=images,
    output_path=pathlib.Path('./Sailor/output')
)
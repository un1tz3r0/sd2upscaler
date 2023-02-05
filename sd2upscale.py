from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline #, StableDiffusionInpaintPipeline
import torch
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any, Callable, Iterable, Sequence, TypeVar, Generic, Type
import math

# ----------------------------------------------------------------------------------------------
# support functions for resizing images and adding margins, and joining and splitting them
# ----------------------------------------------------------------------------------------------

def add_margin(pil_img, top, right, bottom, left, color, mode=None):
		''' add specified numbers of pixels of given color to the edges of the image '''
		width, height = pil_img.size
		new_width = width + right + left
		new_height = height + top + bottom
		result = Image.new(pil_img.mode, (new_width, new_height), color)
		result.paste(pil_img, (left, top))
		return result

def expand2square(pil_img, background_color):
		''' add background_color bars to the edges of the image to make it square'''
		width, height = pil_img.size
		if width == height:
				return pil_img
		elif width > height:
				result = Image.new(pil_img.mode, (width, width), background_color)
				result.paste(pil_img, (0, (width - height) // 2))
				return result
		else:
				result = Image.new(pil_img.mode, (height, height), background_color)
				result.paste(pil_img, ((height - width) // 2, 0))
				return result

def join_images(*images, vertical=False):
		width, height = images[0].width, images[0].height
		tiled_size = (
				(width, height * len(images))
				if vertical
				else (width * len(images), height)
		)
		joined_image = Image.new(images[0].mode, tiled_size)
		row, col = 0, 0
		for image in images:
				joined_image.paste(image, (row, col))
				if vertical:
						col += height
				else:
						row += width
		return joined_image

# ----------------------------------------------------------------------------------------------
# support functions for processing images using overlapping tiles
# ----------------------------------------------------------------------------------------------

def get_tile_start_and_count(image: Image.Image, tilesize: int, tileoverlap: int):
		''' determine the number of tiles vertically and horizontally
		needed to cover the image completely, and the top left corner
		of the first tile so that the tile grid is centered in the image. '''
		w, h = image.size
		x_tiles = math.ceil((w + tileoverlap) / (tilesize - tileoverlap))
		y_tiles = math.ceil((h + tileoverlap) / (tilesize - tileoverlap))
		x_start = (w - (x_tiles * (tilesize - tileoverlap) + tileoverlap)) // 2
		y_start = (h - (y_tiles * (tilesize - tileoverlap) + tileoverlap)) // 2
		return x_start, y_start, x_tiles, y_tiles

def extract_square(
			im: Image.Image, 
			position:Tuple[int, int], 
			size:int, 
			mode=None, 
			background_color=(0,0,0,0)
		):
		''' extract a square of the specified size from the image with top left corner at the specified position.
		if square extends beyond the image bounds, the missing pixels of the returned square are filled with
		background_color, which defaults to black '''
		width, height = im.size
		x, y = position
		# adjust the source rectangle and the destination offset
		source_rectangle = (max(x,0), max(y,0), min(x + size, width), min(y + size, height))
		dest_offset = (max(-x,0), max(-y,0))
		if mode == "RGB":
			square = Image.new('RGB', (size, size), background_color)
		elif mode == "RGBA":
			square = Image.new('RGBA', (size, size), background_color)
		elif mode == None:
			square = Image.new(im.mode, (size, size), background_color)

		square.paste(im.crop(source_rectangle).convert(square.mode), dest_offset)
		return square, source_rectangle, dest_offset

"""
def deprecated(fo):
	def fn(*args, **kwargs):
		raise RuntimeError(f"Call to {fo}({', '.join([*[repr(v) for v in args], *[f'{k}={repr(v)}' for k,v in kwargs]])}) is deprecated. Please update the calling code.")
	return fn

@deprecated
def gaussian_filter(image: np.ndarray, sigma:float):
		''' apply a gaussian filter to the image '''
		return gaussian_filter(image, sigma=sigma, mode='constant', cval=0.0)
"""

def create_tile_mask(size: int, overlap: int):
		from scipy.ndimage import gaussian_filter

		''' create a mask image for use when processing an image using overlapping tiles. the mask is a square image with a gaussian ramp the given number of pixels wide
		at the edges. the mask is returned as a PIL image with mode "L". '''

		if False:
				mask = np.zeros((size, size), dtype=np.float32)
				for x in range(size):
						for y in range(size):
								mask[x,y] = max(0, 1 - math.sqrt((x - size/2)**2 + (y - size/2)**2) / overlap)
		else:
				overlap = overlap // 4
				# generate a square rgb mask the size of a tile with overlap pixel linear gradient border
				mask = np.zeros((size, size), dtype=np.float32)
				#mask[:, :] = 1.0
				mask[overlap:size-overlap, overlap:size-overlap] = 1.0 #np.tile(np.linspace(0, 1, overlap)[:, np.newaxis], size)
				#mask[size-overlap:, :] = 0.0 #np.tile(np.linspace(1, 0, overlap)[:, np.newaxis], size)
				#mask[:, 0:overlap] = 0.0 #np.tile(np.linspace(0, 1, overlap)[np.newaxis, :], (size, 1))
				#mask[:, size-overlap:] = 0.0 #np.tile(np.linspace(1, 0, overlap)[np.newaxis, :], (size, 1))
				mask = gaussian_filter(mask, sigma=overlap/2, mode='constant', cval=0.0)

		# convert to PIL image
		mask = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
		#mask = mask.convert('L')0
		return mask

def paste_square(dest: Image.Image, src: Image.Image, mask: Image.Image, offset: Tuple[int, int]):
		''' pastes the src image over the dest image at the specified offset. all or part of the pasted image may be outside the dest image. if feather > 0, then an
		alpha channel is applied to the src image consisting of a gaussian ramp the given number of pixels wide at the edges of the src image. if processing an image
		using overlapping tiles, feather should be set to the overlap size to enjsure a smooth transition between adjacent tiles.'''

		dw, dh = dest.size
		sw, sh = src.size
		mw, mh = mask.size
		if mw != sw or mh != sh:
				raise ValueError('mask size must match src size')
		dx, dy = offset
		sx, sy = (max(0, -dx), max(0, -dy))
		# adjust the source rectangle and the destination offset
		pw, ph = (max(0, sw-sx), max(0, sh-sy))
		if dx + pw > dw:
				pw = max(dw - dx, 0)
		if dy + ph > dh:
				ph = max(dh - dy, 0)
		dest_offset = (max(dx,0), max(dy,0))
		source_rect = (sx, sy, sx + pw, sy + ph)
		dest_rect = (dest_offset[0], dest_offset[1], dest_offset[0] + pw, dest_offset[1] + ph)
		if pw > 0 and ph > 0:
				return dest.paste(src.crop(source_rect), dest_offset, mask.crop(source_rect))
		return dest

from skimage import exposure

def process_tiles(src, size, overlap, scale, processfunc, progressfunc=None):
		dest = Image.new('RGBA', (src.size[0]*scale, src.size[1]*scale), (0, 0, 0, 0))
		tmask = create_tile_mask(size*scale, overlap*scale)
		x_start, y_start, x_tiles, y_tiles = get_tile_start_and_count(src, size, overlap)
		for y_index in range(y_tiles):
				for x_index in range(x_tiles):
						# find the current tile's top left corner and extract from source
						x = x_start + x_index * (size - overlap)
						y = y_start + y_index * (size - overlap)
						tile, srcrect, tileoff = extract_square(src, (x, y), size)
						srcw, srch = (srcrect[2]-srcrect[0], srcrect[3]-srcrect[1])
						tilex, tiley = tileoff
						tilerect = (tilex, tiley, tilex + srcw, tiley + srch)
						toutrect = (tilerect[0]*scale, tilerect[1]*scale, tilerect[2]*scale, tilerect[3]*scale)

						print(f"processing tile#[{x_index},{y_index}] at ({x}, {y}) of size {size}x{size} with overlap {overlap} and scale {scale} to produce {tile.size[0]*scale}x{tile.size[1]*scale}")

						# run the upscaling pipeline on the source tile
						tout = processfunc(tile)

						# match the output histogram to the non-out-of-bounds area of the source
						tfix = Image.fromarray(exposure.match_histograms(np.asarray(tout.crop(toutrect)), np.asarray(tile.crop(tilerect).resize((srcw*scale, srch*scale))) ))#, channel_axis=2))

						paste_square(dest, tfix, tmask.crop(toutrect), ((tilex+x)*scale, (tiley+y)*scale))

						if progressfunc != None:
								progressfunc(x_index, y_index, x_tiles, y_tiles, (tout, tfix), tmask, dest)
						#dest.show()
		return dest

def upscale_image(source_image, tile_size, tile_overlap, prompt, negative_prompt=None, guidance_scale=7.5, num_inference_steps=10):
		from diffusers import StableDiffusionUpscalePipeline
		# load model and scheduler
		model_id = "stabilityai/stable-diffusion-x4-upscaler"
		pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id) #torch_dtype="auto")
		
		# move to GPU and enable some optimisations
		pipeline = pipeline.to("cuda")
		pipeline.enable_attention_slicing()
		pipeline.enable_xformers_memory_efficient_attention()
		# type: ignore

		#srcimg.show()
		source_image = source_image.convert("RGB")

		def process_tile(tile_image):
				nonlocal pipeline
				nonlocal prompt
				nonlocal negative_prompt
				nonlocal guidance_scale
				nonlocal num_inference_steps
				#tile.show()
				result_image = pipeline(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, image=tile_image, num_inference_steps=num_inference_steps).images[0]
				#result.show()
				return result_image


		def progress(x_index, y_index, x_tiles, y_tiles, tout, tmask, dest):
				print(f"\nprogress: {y_index * x_tiles + x_index}/{y_tiles * x_tiles} processed...\n")

		#with torch.autocast("cuda"):
		upscaled_image = process_tiles(source_image, size=tile_size, overlap=tile_overlap, scale=4, processfunc=process_tile, progressfunc=progress)
		#upscaled_image.show()
		return upscaled_image

# --------------------------------------------------
# demo cli entry point
# --------------------------------------------------

import click
import pathlib
from PIL import Image

@click.command()
@click.argument('source_image', required=True)
@click.argument('output_path', required=True)
@click.option('--prompt', default="", help='The prompt to use for the image.')
@click.option('--negative-prompt', default=None, help='The negative prompt to use for the image.')
@click.option('--tile-size', default=128, help='The size of the tiles to use for processing.')
@click.option('--tile-overlap', default=32, help='The overlap of the tiles to use for processing.')
@click.option('--guidance-scale', default=9.0, help='The scale to use for processing.')
@click.option('--num-inference-steps', default=5, help='The number of steps to use for processing.')
def main(source_image, output_path, prompt, negative_prompt, tile_size, tile_overlap, guidance_scale, num_inference_steps):
		srcimg = Image.open(str(pathlib.Path(source_image).expanduser().resolve()))
		destimg = upscale_image(source_image=srcimg, tile_size=tile_size, tile_overlap=tile_overlap, prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
		destimg.save(str(pathlib.Path(output_image).expanduser().resolve()))



if __name__ == '__main__':
		main()

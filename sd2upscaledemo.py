from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Union, List, Tuple, Optional, Dict, Any, Callable, Iterable, Sequence, TypeVar, Generic, Type
import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------
# options data classes 

@dataclass
class PipelineOptions:
	''' options for pipeline '''
	prompt:Optional[str] = None
	negative_prompt:Optional[str] = None
	num_inference_steps:Optional[int] = None
	guidance_scale:Optional[float] = None
	strength:Optional[float] = None
	add_predicted_noise:Optional[bool] = None
	noise_level:Optional[float] = None

default_outpaint_options = PipelineOptions(
	num_inference_steps=25, 
	guidance_scale=7.5, 
	prompt="wide-angle", 
	negative_prompt="text, label, logo, sign, poor quality, letters, characters, symbols, numbers, blurry, noisy, out of focus, low quality, artifacts, compressed, digits, writing, headline, title, heading, ad, banner, promo"
)

default_upscaler_options = PipelineOptions(
	num_inference_steps=5, 
	guidance_scale=9.0, 
	prompt="detailed, intricate", 
	negative_prompt="text, label, logo, sign, poor quality, letters, characters, symbols, numbers, blurry, noisy, out of focus, low quality, artifacts, compressed, digits, writing, headline, title, heading, ad, banner, promo"
)

def get_pipeline_options_dict(*options_args:List[PipelineOptions]):
	''' get a dictionary of pipeline options by combining the options from the specified options objects, with the first object's values taking precedence '''
	def get_option(name:str):
		for options_arg in options_args:
			if options_arg == None:
				continue
			if getattr(options_arg, name, None) != None:
				return getattr(options_arg, name)
		return None
	result = {}
	for name in ['prompt', 'negative_prompt', 'num_inference_steps', 'guidance_scale', 'strength', 'add_predicted_noise', 'noise_level']:
		if get_option(name) != None:
			result[name] = get_option(name)
	return result


# ----------------------------------------------------------------------------------------------
# support functions for resizing images and adding margins, and joining and splitting them
# ----------------------------------------------------------------------------------------------

def add_margin(img:Image.Image, left, top, right, bottom, color, mode=None, fill=True):
		''' add specified numbers of pixels of given color to the edges of the image '''
		if mode is None:
				mode = img.mode
		width, height = img.size
		new_width = width + right + left
		new_height = height + top + bottom
		result = Image.new(mode, (new_width, new_height), color)
		result.paste(img, (left, top))
		if fill:
			# fill corners
			if left > 0 and top > 0:
				result.paste(img.crop((0,0,1,1)).resize((left,top)), (0,0))
			if right > 0 and top > 0:
				result.paste(img.crop((width-1,0,width,height)).resize((right,top)), (new_width-right,0))
			if left > 0 and bottom > 0:
				result.paste(img.crop((0,height-1,width,height)).resize((left,bottom)), (0,new_height-bottom))
			if right > 0 and bottom > 0:
				result.paste(img.crop((width-1,height-1,width,height)).resize((right,bottom)), (new_width-right,new_height-bottom))
			# fill edges
			if top > 0:
				result.paste(img.crop((0,0,width,1)).resize((width,top)), (left,0))
			if bottom > 0:
				result.paste(img.crop((0,height-1,width,height)).resize((width,bottom)), (left,new_height-bottom))
			if left > 0:
				result.paste(img.crop((0,0,1,height)).resize((left,height)), (0,top))
			if right > 0:
				result.paste(img.crop((width-1,0,width,height)).resize((right,height)), (new_width-right,top))
		return result

def outpaint_mask(width, height, left, top, right, bottom):
	''' create a mask for outpainting the specified region of an image '''
	mask = np.zeros((height, width), dtype=np.float32)
	mask[top:height-bottom, left:width-right] = 1.0
	#mask = gaussian_filter(mask, sigma=5)
	#mask = np.clip(mask, 0, 1)
	return Image.fromarray((mask*255.0).astype(dtype=np.int8), mode="L")

def outpaint_horizontal(img:Image.Image, add_pixels_left=100, add_pixels_right=100, options:PipelineOptions=None, left_options:PipelineOptions=None, right_options:PipelineOptions=None):
	''' outpaint the left and right of an image, increasing its height by the specified number of pixels on each side. 
		Note that the image will be resized to a square region for inpainting, and downscaled to 512x512 for processing,
		after which the inpainted regions will be upscaled to the original size and joined to the original image. '''
	
	# ratio of left padding to total padding
	left_ratio = add_pixels_left / (add_pixels_left + add_pixels_right)
	right_ratio = add_pixels_right / (add_pixels_left + add_pixels_right)

	# calculate padding and cropping amounts to get square images for inpainting
	add_adjusted_left = add_pixels_left
	add_adjusted_right = add_pixels_right
	crop_after_left = 0
	crop_after_right = 0
	scale_factor = 1.0
	if img.width + add_adjusted_left + add_adjusted_right < img.height:
		add_adjusted = img.height - img.width
		add_adjusted_left = math.round(add_adjusted * left_ratio)
		add_adjusted_right = math.round(add_adjusted * right_ratio)
		crop_after_left = add_adjusted_left - add_pixels_left
		crop_after_right = add_adjusted_right - add_pixels_right
	
	# pad and crop image to square regions for inpainting (left and right)
	padded_img = add_margin(img, add_adjusted_left, 0, add_adjusted_right, 0, color=(0,0,0,0), mode="RGBA")
	croppedleft_img = padded_img.crop((0, 0, padded_img.height, padded_img.height))
	croppedright_img = padded_img.crop((padded_img.width - padded_img.height, 0, padded_img.width, padded_img.height))
	# downscale inpaint input squares from image to 512x512
	scale_factor = 512 / padded_img.height
	scaledleft_img = croppedleft_img.resize((512, 512), resample=Image.BICUBIC)
	scaledright_img = croppedright_img.resize((512, 512), resample=Image.BICUBIC)

	# extract alpha channels for inpainting masks
	def alpha_to_mask(image):
		alpha = image.split()[-1]
		mask = Image.new("RGBA", alpha.size, (0,0,0,255))
		mask.paste(alpha, mask=alpha)
		mask = mask.point(lambda x: 0 if x>0 else 255)
		mask = mask.convert("L")
		return mask
	
	# create inpainting masks with scaled, cropped padding blacked out
	scaledleftmask_img = outpaint_mask(512, 512, math.ceil(scale_factor * add_adjusted_left), 0, 0, 0) #alpha_to_mask(scaledleft_img)
	scaledrightmask_img = outpaint_mask(512, 512, 0, 0, math.ceil(scale_factor * add_adjusted_right), 0) #alpha_to_mask(scaledright_img)

	# load model
	pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
	pipeline.enable_attention_slicing()
	pipeline.enable_xformers_memory_efficient_attention()
	pipeline.to("cuda")
	
	# run the pipeline
	left_out = pipeline(image=scaledleft_img, mask_image=scaledleftmask_img, **get_pipeline_options_dict(left_options, options, default_outpaint_options)).images[0]
	right_out = pipeline(image=scaledright_img, mask_image=scaledrightmask_img, **get_pipeline_options_dict(right_options, options, default_outpaint_options)).images[0]
	
	# upscale the outpainted image
	unscaledleft_out = left_out.resize((croppedleft_img.width, croppedleft_img.height), resample=Image.BICUBIC)
	unscaledright_out = right_out.resize((croppedright_img.width, croppedright_img.height), resample=Image.BICUBIC)
	
	# paste the outpainted parts back into the padded original image
	padded_img.paste(unscaledleft_out.crop((0,0,add_adjusted_left,unscaledleft_out.height)), (0, 0))
	padded_img.paste(unscaledright_out.crop((unscaledright_out.width-add_adjusted_right,0,unscaledright_out.width,unscaledright_out.height)), (padded_img.width - add_adjusted_right, 0))

	# remove extra padding added to make the image square
	padded_img = padded_img.crop((crop_after_left, 0, padded_img.width - crop_after_right, padded_img.height))

	return padded_img	
	

def outpaint_vertical(img:Image.Image, add_pixels_top=100, add_pixels_bottom=100, options:PipelineOptions=None,	top_options:PipelineOptions=None, bottom_options:PipelineOptions=None):
	''' outpaint the top and bottom of an image, increasing its height by the specified number of pixels on each side. 
		Note that the image will be resized to a square region for inpainting, and downscaled to 512x512 for processing,
		after which the inpainted regions will be upscaled to the original size and joined to the original image. '''
	
	# ratio of top padding to total padding
	top_ratio = add_pixels_top / (add_pixels_top + add_pixels_bottom)
	bottom_ratio = add_pixels_bottom / (add_pixels_top + add_pixels_bottom)

	# calculate padding and cropping amounts to get square images for inpainting
	add_adjusted_top = add_pixels_top
	add_adjusted_bottom = add_pixels_bottom
	crop_after_top = 0
	crop_after_bottom = 0
	scale_factor = 1.0
	if img.height + add_adjusted_top + add_adjusted_bottom < img.width:
		add_adjusted = img.width - img.height
		add_adjusted_top = math.round(add_adjusted * top_ratio)
		add_adjusted_bottom = math.round(add_adjusted * bottom_ratio)
		crop_after_top = add_adjusted_top - add_pixels_top
		crop_after_bottom = add_adjusted_bottom - add_pixels_bottom
	
	# pad and crop image to square regions for inpainting (top and bottom)
	padded_img = add_margin(img, 0, add_adjusted_top, 0, add_adjusted_bottom, color=(0,0,0,0), mode="RGBA")
	croppedtop_img = padded_img.crop((0, 0, padded_img.width, padded_img.width))
	croppedbottom_img = padded_img.crop((0, padded_img.height - padded_img.width, padded_img.width, padded_img.width))
	# downscale inpaint input squares from image to 512x512
	scale_factor = 512 / padded_img.width
	scaledtop_img = croppedtop_img.resize((512, 512), resample=Image.BICUBIC)
	scaledbottom_img = croppedbottom_img.resize((512, 512), resample=Image.BICUBIC)

	# extract alpha channels for inpainting masks
	def alpha_to_mask(image):
		alpha = image.split()[-1]
		mask = Image.new("RGBA", alpha.size, (0,0,0,255))
		mask.paste(alpha, mask=alpha)
		mask = mask.point(lambda x: 0 if x>0 else 255)
		mask = mask.convert("L")
		return mask
	
	# create inpainting masks with scaled, cropped padding blacked out
	scaledtopmask_img = outpaint_mask(512, 512, 0, math.ceil(scale_factor * add_adjusted_top), 0, 0) #alpha_to_mask(scaledleft_img)
	scaledbottommask_img = outpaint_mask(512, 512, 0, 0, 0, math.ceil(scale_factor * add_adjusted_bottom)) #alpha_to_mask(scaledright_img)

	# load model
	pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
	pipeline.enable_attention_slicing()
	pipeline.enable_xformers_memory_efficient_attention()
	pipeline.to("cuda")
	
	# run the pipeline
	top_out = pipeline(image=scaledtop_img, mask_image=scaledtopmask_img, **get_pipeline_options_dict(top_options, options, default_outpaint_options)).images[0]
	bottom_out = pipeline(image=scaledbottom_img, mask_image=scaledbottommask_img, **get_pipeline_options_dict(bottom_options, options, default_outpaint_options)).images[0]
	
	# upscale the outpainted image
	unscaledtop_out = top_out.resize((croppedtop_img.width, croppedtop_img.height), resample=Image.BICUBIC)
	unscaledbottom_out = bottom_out.resize((croppedbottom_img.width, croppedbottom_img.height), resample=Image.BICUBIC)
	
	# paste the outpainted parts back into the padded original image
	padded_img.paste(unscaledtop_out.crop((0,0,unscaledtop_out.width,add_adjusted_top)), (0, 0))
	padded_img.paste(unscaledbottom_out.crop((0,unscaledbottom_out.height-add_adjusted_bottom,unscaledbottom_out.width,unscaledbottom_out.height)), (0, padded_img.height - add_adjusted_bottom))

	# remove extra padding added to make the image square
	padded_img = padded_img.crop((0, crop_after_top, padded_img.width, padded_img.height - crop_after_bottom))

	return padded_img	

# ----------------------------------------------------------------------------------------------
# support functions for processing images using overlapping tiles
# ----------------------------------------------------------------------------------------------

def get_tile_start_and_count(image, tilesize, tileoverlap):
		''' determine the number of tiles vertically and horizontally
		needed to cover the image completely, and the top left corner
		of the first tile so that the tile grid is centered in the image. '''
		w, h = image.size
		x_tiles = math.ceil((w + tileoverlap) / (tilesize - tileoverlap))
		y_tiles = math.ceil((h + tileoverlap) / (tilesize - tileoverlap))
		x_start = (w - (x_tiles * (tilesize - tileoverlap) + tileoverlap)) // 2
		y_start = (h - (y_tiles * (tilesize - tileoverlap) + tileoverlap)) // 2
		return x_start, y_start, x_tiles, y_tiles

def extract_square(im, position:Tuple[int, int], size:int, mode=None, background_color=(0,0,0,0)):
		''' extract a square of the specified size from the image with top left corner at the specified position.
		if square extends beyond the image bounds, the missing pixels of the returned square are filled with
		background_color, which defaults to black '''
		#with Image.open(image_path) as im:
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

def create_tile_mask(size: int, overlap: int):

		''' create a mask image for use when processing an image using overlapping tiles. the mask is a square image with a gaussian ramp the given number of pixels wide
		at the edges. the mask is returned as a PIL image with mode "L". '''

		# generate a square rgb mask the size of a tile with overlap pixel linear gradient border
		overlap = overlap // 4
		mask = np.zeros((size, size), dtype=np.float32)
		mask[overlap:size-overlap, overlap:size-overlap] = 1.0
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
		# create a buffer for the output image of src's size multiplied by scale
		dest = Image.new('RGBA', (src.size[0]*scale, src.size[1]*scale), (0, 0, 0, 0))
		# create the mask which we will use to smoothly blend the edges of the output tiles
		tmask = create_tile_mask(size*scale, overlap*scale)
		# calculate the tile grid offset and dimensions nescessary to cover the source image
		x_start, y_start, x_tiles, y_tiles = get_tile_start_and_count(src, size, overlap)
		# iterate the tiles
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
						outrect = (srcrect[0]*scale, srcrect[1]*scale, srcrect[2]*scale, srcrect[3]*scale)

						if progressfunc == None:
							print(f"[upscaler] processing tile [{x_index},{y_index}] at ({x}, {y}) of size {size}x{size} with overlap {overlap} and scale {scale} to produce {tile.size[0]*scale}x{tile.size[1]*scale}")

						# run the upscaling pipeline on the source tile
						tout = processfunc(tile)

						# match the output histogram to the non-out-of-bounds area of the source
						tfix = Image.fromarray(exposure.match_histograms(np.asarray(tout.crop(toutrect)), np.asarray(tile.crop(tilerect).resize((srcw*scale, srch*scale))), channel_axis=2))

						paste_square(dest, tfix, tmask.crop(toutrect), ((tilex+x)*scale, (tiley+y)*scale))

						if progressfunc != None:
								progressfunc(x_index, y_index, x_tiles, y_tiles, 1, 1, dest, outrect)
						#dest.show()
		return dest

def upscale_image(source_image, tile_size=128, tile_overlap=48, pipeline_options:PipelineOptions=None, progressfunc=None, updatefunc=None):
		default_options = PipelineOptions(num_inference_steps=5, guidance_scale=9.0, prompt="detailed, 8k, high quality, hd", negative_prompt="blurry, noisy, out of focus, low quality, artifacts, compressed")
		
		from diffusers import StableDiffusionUpscalePipeline
		
		#  load model and scheduler
		model_id = "stabilityai/stable-diffusion-x4-upscaler"
		pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id) # type: StableDiffusionUpscalePipeline
		pipeline.enable_attention_slicing()
		pipeline.enable_xformers_memory_efficient_attention()
		pipeline = pipeline.to("cuda")

		#srcimg.show()
		source_image = source_image.convert("RGB")

		def progress(x_index, y_index, x_tiles, y_tiles, current_step, total_steps, dest_image, dest_rect):
				print(f"\nprogress: tile {y_index * x_tiles + x_index} of {y_tiles * x_tiles}\n")
				if updatefunc != None:
						updatefunc(dest_image, dest_rect)
		
		def process_tile(tile_image):
				nonlocal pipeline
				#tile.show()
				result_image = pipeline(image=tile_image, **get_pipeline_options_dict(pipeline_options, default_options)).images[0]
				#result.show()
				return result_image

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
@click.argument('dest_image', required=True)
@click.option('--prompt', default="", help='The prompt to use for the image.')
@click.option('--negative-prompt', default=None, help='The negative prompt to use for the image.')
@click.option('--tile-size', default=128, help='The size of the tiles to use for processing.')
@click.option('--tile-overlap', default=48, help='The overlap of the tiles to use for processing.')
@click.option('--guidance-scale', default=9.0, help='The guidance scale to use for processing.')
@click.option('--num-inference-steps', default=10, help='The number of steps to use for processing.')
@click.option('--add-predicted-noise', default=None, help='The noise level to use for processing.')
@click.option('--noise-level', default=None, help='The noise level to use for processing.')
def main(source_image, dest_image, *, prompt, negative_prompt, tile_size, tile_overlap, scale, steps, add_predicted_noise, noise_level, eta):
		srcimg = Image.open(str(pathlib.Path(source_image).expanduser().resolve()))
		def updatefunc(im, rect):
				if rect != None:
					outim = im.copy()
					rl, rt, rr, rb = rect
					import PIL.ImageDraw
					draw = PIL.ImageDraw.Draw(im)
					draw.rectangle((rl, rt, rr, rb), outline="red", width=10)
					outim.save(str(pathlib.Path(dest_image).expanduser().resolve()))
		uim = upscale_image(source_image=srcimg, tile_size=tile_size, tile_overlap=tile_overlap, prompt=prompt, negative_prompt=negative_prompt, guidance_scale=scale, steps=steps, updatefunc=updatefunc)
		uim.save(str(pathlib.Path(dest_image).expanduser().resolve()))

if __name__ == '__main__':
		main()


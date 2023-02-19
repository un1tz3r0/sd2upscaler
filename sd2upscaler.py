from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure # for correcting color drift when upscaling tiles
from typing import Union, List, Tuple, Optional, Dict, Any, Callable, Iterable, Sequence, TypeVar, Generic, Type
from types import NoneType

import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------
# options data class
# ----------------------------------------------------------------------------------------------

@dataclass
class PipelineOptions:
	''' options for stable diffusion inpainting and upscaling pipelines '''
	prompt :Optional[str] = None
	negative_prompt :Optional[str] = None
	num_inference_steps :Optional[int] = None
	guidance_scale :Optional[float] = None
	strength :Optional[float] = None
	add_predicted_noise :Optional[bool] = None
	noise_level :Optional[float] = None
	eta :Optional[float] = None


# ----------------------------------------------------------------------------------------------
# named pipeline options profiles for quick configuration
# ----------------------------------------------------------------------------------------------

default_outpaint_options = PipelineOptions(
	num_inference_steps=25,
	guidance_scale=9.0,
	prompt=None, # "wide-angle, wide-view, panoramic, seamless, endless, photosphere",
	negative_prompt=None # "text, label, logo, sign, watermark, poor quality, letters, characters, symbols, numbers, blurry, noisy, out of focus, low quality, artifacts, compressed, digits, writing, headline, title, heading, ad, banner, promo"
)

default_upscaler_options = PipelineOptions(
	num_inference_steps=10,
	guidance_scale=7.5,
	prompt=None, # "detailed, intricate, 8k, hdri, high quality, professional, gigapixel, retina",
	negative_prompt=None # "text, label, logo, sign, poor quality, letters, characters, symbols, numbers, blurry, noisy, out of focus, low quality, artifacts, compressed, digits, writing, headline, title, heading, ad, banner, promo"
)

def get_pipeline_options_dict(*options_args:List[PipelineOptions]):
	''' get a dictionary of pipeline options by combining the options from the
	specified options objects, with the first object whose value is not None
	taking precedence '''

	def get_option(name:str):
		for options_arg in options_args:
			if options_arg == None:
				continue
			if getattr(options_arg, name, None) != None:
				return getattr(options_arg, name)
		return None

	result = {}
	for name in PipelineOptions.__dataclass_fields__.keys():
		if get_option(name) != None:
			result[name] = get_option(name)
	return result

'''
# above, as a one-liner ;)
options = PipelineOptions()
def as_dict(*opts_args:List[PipelineOptions]):
	return [k: v for k,v in {
			key: (
					[
							opts[key] for opts in opts_args
								if key in vars(opts).keys() and vars(opts)[key] != None
					]+[
							None

					]
			)[0] for key in PipelineOptions.__dataclass_fields__
		}.items() if v != None
	]
'''

# ----------------------------------------------------------------------------------------------
# support functions for resizing images and adding margins, and joining and splitting them
# ----------------------------------------------------------------------------------------------

def add_margin(img:Image.Image, left, top, right, bottom, color=None, mode=None, fill=True, fade_to_color=None, fade_edge_alpha=0.0):
		''' add specified numbers of pixels of given color to the edges of the image. if fill is False, margin will be filled with color,
		which defaults to black with fully transparent alpha. if fill is true, margin defaults to pixel color at mnearestedge of image '''

		if mode == None:
			# use same mode as input image by default
			mode = img.mode

		if color == None:
			# use black, transparent by default
			if mode == "RGB":
				color = (0,0,0)
			if mode == "RGBA":
				color = (0,0,0,0)
			if mode == "L":
				color = (0,)
			if mode == "LA":
				color = (0,0)

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
		
		#result.show()
		return result

# extract alpha channels for inpainting masks
def alpha_to_mask(image):
	''' unused, but useful later maybe, extract alpha channel from image and convert to mask '''
	alpha = image.split()[-1]
	mask = Image.new("RGBA", alpha.size, (0,0,0,255))
	mask.paste(alpha, mask=alpha)
	mask = mask.point(lambda x: 0 if x>0 else 255)
	mask = mask.convert("L")
	return mask

def create_outpaint_mask(width, height, left, top, right, bottom, 
			 feather_left=0, feather_top=0, feather_right=0, feather_bottom=0, invert=True):
	''' create a mask for outpainting the specified region of an image '''

	@dataclass
	class Rect:
		left: int
		top: int
		right: int
		bottom: int

	mask = np.zeros((height, width), dtype=np.float32)
	mask[top:height-bottom, left:width-right] = 1.0

	# feather inward from inner-outer boundary
	oh = height - top - bottom
	ow = width - left - right
	ih = oh - feather_bottom - feather_top
	iw = ow - feather_left - feather_right
	orect = Rect(left, top, width-right, height-bottom)
	irect = Rect(left+feather_left, top+feather_top, width-right-feather_right, height-bottom-feather_bottom)
	if feather_top > 0:
		mask[orect.top:irect.top, :] *= np.tile(np.linspace(0, 1, irect.top-orect.top)[:, np.newaxis], (1, width))
	if feather_bottom > 0:
		mask[irect.bottom:orect.bottom, :] *= np.tile(np.linspace(1, 0, orect.bottom-irect.bottom)[:, np.newaxis], (1, width))
	if feather_left > 0:
		mask[:, orect.left:irect.left] *= np.tile(np.linspace(0, 1, irect.left-orect.left)[np.newaxis, :], (height, 1))
	if feather_right > 0:
		mask[:, irect.right:orect.right] *= np.tile(np.linspace(1, 0, orect.right-irect.right)[np.newaxis, :], (height, 1))
	
	if invert:
		mask = 1.0 - mask

	return Image.fromarray((mask*255.0).astype(dtype=np.int8), mode="L")



def outpaint_horizontal(img:Image.Image, add_pixels_left:int=0, add_pixels_right:int=0, feather_pixels_left=20, feather_pixels_right=20,  
			options:Optional[PipelineOptions]=None, left_options:Optional[PipelineOptions]=None, right_options:Optional[PipelineOptions]=None, 
			fill_only=False):
	''' expand img horizontally by adding `add_pixels_left` and `add_pixels_right` width borders to the left and right
		of the source image, increasing its width by the specified number of pixels on each side. The largest square
		are fit at the left and right of the new image boundary, and then scaled down to 512x512 hat the image will be resized to a square region for inpainting, and downscaled to 512x512 for processing,
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
		add_adjusted_left = math.floor(add_adjusted * left_ratio)
		add_adjusted_right = math.ceil(add_adjusted * right_ratio)
		crop_after_left = add_adjusted_left - add_pixels_left
		crop_after_right = add_adjusted_right - add_pixels_right

	# pad and crop image to square regions for inpainting (left and right)
	padded_img = add_margin(img, add_adjusted_left, 0, add_adjusted_right, 0, color=(0,0,0,0), mode="RGBA")

	if not fill_only:
		croppedleft_img = padded_img.crop((0, 0, padded_img.height, padded_img.height))
		croppedright_img = padded_img.crop((padded_img.width - padded_img.height, 0, padded_img.width, padded_img.height))
		# downscale inpaint input squares from image to 512x512
		scale_factor = 512 / padded_img.height
		scaledleft_img = croppedleft_img.resize((512, 512), resample=Image.BICUBIC)
		scaledright_img = croppedright_img.resize((512, 512), resample=Image.BICUBIC)

		# create inpainting masks with scaled, cropped padding blacked out
		scaledleftmask_img = create_outpaint_mask(512, 512, math.ceil(scale_factor * add_adjusted_left), 0, 0, 0) #alpha_to_mask(scaledleft_img)
		scaledrightmask_img = create_outpaint_mask(512, 512, 0, 0, math.ceil(scale_factor * add_adjusted_right), 0) #alpha_to_mask(scaledright_img)

		# load model
		pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
		pipeline.enable_attention_slicing() # type: ignore
		pipeline.enable_xformers_memory_efficient_attention() # type: ignore
		pipeline.to("cuda") # type: ignore

		# run the pipeline(s) to inpaint the left and right sides
		left_out = pipeline(image=scaledleft_img, mask_image=scaledleftmask_img, **get_pipeline_options_dict(left_options, options, default_outpaint_options)).images[0]
		right_out = pipeline(image=scaledright_img, mask_image=scaledrightmask_img, **get_pipeline_options_dict(right_options, options, default_outpaint_options)).images[0]

		# upscale the outpainted image
		unscaledleft_out = left_out.resize((croppedleft_img.width, croppedleft_img.height), resample=Image.BICUBIC)
		unscaledright_out = right_out.resize((croppedright_img.width, croppedright_img.height), resample=Image.BICUBIC)

	  # create recombining masks with feathering to blend inpainted regions with original image
		unscaledleft_mask = create_outpaint_mask(add_adjusted_left+feather_pixels_left,unscaledleft_out.height, add_adjusted_left, 0, 0, 0, feather_left=feather_pixels_left, invert=True) #alpha_to_mask(scaledleft_img)
		unscaledright_mask = create_outpaint_mask(add_adjusted_right+feather_pixels_right, unscaledright_out.height, 0, 0, add_adjusted_right, 0, feather_right=feather_pixels_right, invert=True) #alpha_to_mask(scaledright_img)

		# paste the outpainted parts back into the padded original image
		padded_img.paste(unscaledleft_out.crop((0,0,add_adjusted_left+feather_pixels_left,unscaledleft_out.height)), (0, 0), mask=unscaledleft_mask)
		padded_img.paste(unscaledright_out.crop((unscaledright_out.width-(add_adjusted_right+feather_pixels_right),0,unscaledright_out.width,unscaledright_out.height)), (padded_img.width - (add_adjusted_right + feather_pixels_right), 0), mask=unscaledright_mask)

	# remove extra padding added to make the image square
	padded_img = padded_img.crop((crop_after_left, 0, padded_img.width - crop_after_right, padded_img.height))

	return padded_img


def outpaint_vertical(img:Image.Image, add_pixels_top=100, add_pixels_bottom=100, feather_pixels_top=20, feather_pixels_bottom=20,
		      options:PipelineOptions=None,	top_options:PipelineOptions=None, bottom_options:PipelineOptions=None,
		      fill_only=False):
	''' outpaint the top and bottom of an image, increasing its height by the specified number of pixels on each side.
		Note that the image will be resized to a square region for inpainting, and downscaled to 512x512 for processing,
		after which the inpainted regions will be upscaled to the original size and joined to the original image. '''

  # FIXME: add more sanity checks and i.e. non-negative pixel counts

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
		add_adjusted_top = math.floor(add_adjusted * top_ratio)
		add_adjusted_bottom = math.ceil(add_adjusted * bottom_ratio)
		crop_after_top = add_adjusted_top - add_pixels_top
		crop_after_bottom = add_adjusted_bottom - add_pixels_bottom

	# pad and crop image to square regions for inpainting (square spanning whoe width, justified at top and bottom)
	padded_img = add_margin(img, 0, add_adjusted_top, 0, add_adjusted_bottom, color=(0,0,0,0), mode="RGBA")

	if not fill_only:
		croppedtop_img = padded_img.crop((0, 0, padded_img.width, padded_img.width))
		croppedbottom_img = padded_img.crop((0, padded_img.height - padded_img.width, padded_img.width, padded_img.height))
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
		scaledtopmask_img = create_outpaint_mask(512, 512, 0, math.ceil(scale_factor * add_adjusted_top), 0, 0) #alpha_to_mask(scaledleft_img)
		scaledbottommask_img = create_outpaint_mask(512, 512, 0, 0, 0, math.ceil(scale_factor * add_adjusted_bottom)) #alpha_to_mask(scaledright_img)

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
	  
	  # create recombining masks with feathering to blend inpainted regions with original image
		unscaledtop_mask = create_outpaint_mask(unscaledtop_out.width, add_adjusted_top+feather_pixels_top, 0, add_adjusted_top, 0, 0, feather_top=feather_pixels_top, invert=True) #alpha_to_mask(scaledleft_img)
		unscaledbottom_mask = create_outpaint_mask(unscaledbottom_out.width, add_adjusted_bottom+feather_pixels_bottom, 0, 0, 0, add_adjusted_bottom, feather_bottom=feather_pixels_bottom, invert=True) #alpha_to_mask(scaledright_img)

		# paste the outpainted parts back into the padded original image
		padded_img.paste(unscaledtop_out.crop((0,0,unscaledtop_out.width,add_adjusted_top+feather_pixels_top)), (0, 0), mask=unscaledtop_mask)
		padded_img.paste(unscaledbottom_out.crop((0,unscaledbottom_out.height-(add_adjusted_bottom+feather_pixels_bottom),unscaledbottom_out.width,unscaledbottom_out.height)), (0, padded_img.height - (add_adjusted_bottom + feather_pixels_bottom)), mask=unscaledbottom_mask)

	# remove extra padding added to make the image square
	padded_img = padded_img.crop((0, crop_after_top, padded_img.width, padded_img.height - crop_after_bottom))

	return padded_img


# ----------------------------------------------------------------------------------------------
# support functions for progressively outpainting images using overlapping tiles
# ----------------------------------------------------------------------------------------------

"""
# extra broken rn...

def outpaint_side_tiled(img, pipeline_options, tile_size=512, tile_u_overlap=256, tile_v_overlap=256, side="right", pixels=1024):
	''' u -> perpendicular to side being outpainted
			v -> parallel to side being outpainted.
	
	outpaints the given side by pixels using overlapping tiles 
	'''
	from math import ceil, floor, remainder, trunc

	''' we define the area to be outpainted and the direction in which to march along it to fill it with tiles, based on the requested side and lengthwise and crosswise overlap. '''
	if side == "left" or side == "right":
		du = tile_size - tile_u_overlap
		dv = tile_size - tile_v_overlap
		startx = 0 if side == "left" else img.width - tile_size
		starty = 0
		ymax = img.height
		xmax = img.width - tile_size
		xincr = (du if side == "left" else -du, 0) # du, dv
		yincr = (0, dv) # du, dv

	elif side == "bottom" or side == "top":
		du = tile_size - tile_v_overlap
		dv = tile_size - tile_u_overlap
		startx = 0
		starty = 0 if side == "top" else img.height - tile_size
		ymax = img.height - tile_size
		xmax = img.width
		xincr = (0, dv)
		yincr = (du if side == "top" else -du, 0)

	else:
		raise ValueError("side must be one of 'left', 'right', 'top', 'bottom'")

	# create the outpaint mask
	mask = create_outpaint_mask(img.width, img.height, startx, startx + pixels, starty, starty + pixels)

	# run the pipeline
	out = pipeline(image=img, mask_image=mask, **pipeline_options).images[0]

	# paste the outpainted parts back into the padded original image
	img.paste(out.crop((startx, starty, startx + pixels, starty + pixels)), (startx, starty), mask=mask.crop((startx, starty, startx + pixels, starty + pixels)))

	# fill in the rest of the side
	for y in range(starty, ymax, dv):
		for x in range(startx, xmax, du):
			# create the outpaint mask
			mask = create_outpaint_mask(img.width, img.height, x, x + tile_size, y, y + tile_size)

			# run the pipeline
			out = pipeline(image=img, mask_image=mask, **pipeline_options).images[0]

			# paste the outpainted parts back into the padded original image
			img.paste(out.crop((x, y, x + tile_size, y + tile_size)), (x, y), mask=mask.crop((x, y, x + tile_size, y + tile_size)))

	return img

	du = tile_size - tile_u_overlap
	dv = tile_size - tile_v_overlap
	if side == "left" or side == "right":
		startx = 0 if side == "left" else img.width - tile_size
		starty = 0
		ymax = img.height
		xmax = 
		xincr = (du if side == "left" else -du, 0) # du, dv
		yincr = (0, dv) # du, dv

	elif side == "bottom" or side == "top":
		startx = 0
		starty = 0 if side == "top" else img.height - tile_size
		xincr = (0, dv)
		yincr = (du if side == "top" else -du, 0)
"""

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

def create_tile_mask(size: int, overlap: int, featherleft: bool = True, feathertop: bool = True, featherright: bool = False, featherbottom: bool = False,  feather_top: int = 0, feather_bottom: int = 0, feather_left: int = 0, feather_right: int = 0, invert: bool = False):

		''' create a mask image for use when processing an image using overlapping tiles. the mask is a square image with a
		ramp the given number of pixels wide on the specified edges. the mask is returned as a PIL image with mode "L". '''

		if True:
			# generate a square rgb mask the size of a tile with overlap pixel linear gradient border

			mask = np.zeros((size, size, 3), dtype=np.float32)
			mask[:, :, :] = 1.0
			if feathertop:
				mask[0:(overlap), :, :] = np.repeat(np.tile(np.linspace(0, 1, overlap)[:, np.newaxis], size)[:, :, np.newaxis], 3, axis=2)
			if featherbottom:
				mask[((size-overlap)):(size), :, :] = np.repeat(np.tile(np.linspace(1, 0, overlap)[:, np.newaxis], size)[:, :, np.newaxis], 3, axis=2)
			if featherleft:
				mask[:, 0:(overlap), :] *= np.repeat(np.tile(np.linspace(0, 1, overlap)[np.newaxis, :], (size, 1))[:, :, np.newaxis], 3, axis=2)
			if featherright:
				mask[:, ((size-overlap)):(size), :] *= np.repeat(np.tile(np.linspace(1, 0, overlap)[np.newaxis, :], (size, 1))[:, :, np.newaxis], 3, axis=2)

		else:
			# generate a square rgb mask the size of a tile with overlap pixel linear gradient border
			overlap = overlap // 2
			mask = np.zeros((size, size), dtype=np.float32)
			#mask[overlap:size-overlap, overlap:size-overlap] = 1.0
			mask[0:overlap, :] = np.tile(np.linspace(0.0, 1.0, overlap, dtype=np.float32)[:, np.newaxis], size)
			mask[:, :] = 1.0
			mask = gaussian_filter(mask, sigma=overlap, mode='constant', cval=0.0)

		# convert to PIL image
		mask = Image.fromarray((mask * 255.0).astype(np.uint8), mode="RGB")
		mask = mask.convert('L')
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
				srccrop = src.crop(source_rect)
				maskcrop = mask.crop(source_rect)
				srccrop.putalpha(maskcrop)
				return dest.alpha_composite(srccrop, dest_offset)
		return dest


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

def upscale_image(source_image, tile_size=128, tile_overlap=32, pipeline_options:PipelineOptions=None, progressfunc=None, updatefunc=None):
		default_options = default_upscaler_options

		from diffusers import StableDiffusionUpscalePipeline

		#  load model and scheduler
		model_id = "stabilityai/stable-diffusion-x4-upscaler"
		pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id) # type: ignore # type: StableDiffusionUpscalePipeline
		pipeline.enable_attention_slicing() # type: ignore
		pipeline.enable_xformers_memory_efficient_attention() # type: ignore
		pipeline = pipeline.to("cuda") # type: ignore

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
import pathlib, math
from PIL import Image

@click.command("sd2upscale")
@click.argument('source_images', nargs=-1, type=click.Path())
@click.option('--dest-dir', default=None, help='The directory to save the output images to. If not specified, the output images will be saved to the same directory as the source image.')
@click.option('--dest-name', default=None, help='Name template for output images. Substiutions: {parent} parent dir of source image, {name} name of source image, {ext} extension of source image. Default: {name}-sd2out{ext}')
#@click.group(name="Outpainting")
@click.option("--prompt", default=None, help="The default prompt to use.")
@click.option("--negative-prompt", default=None, help="The default negative prompt to use.")
@click.option('--outpaint-aspect-ratio', default=None, help='Outpaint the image to the specified aspect ratio (overrides --outpaint-top, --outpaint-bottom, --outpaint-left, --outpaint-right)')
@click.option('--outpaint-fill-only', default=False, is_flag=True, help='Output the pre-processed image normally fed through the SD2 inpainting model as-is. (overrides --outpaint-top, --outpaint-bottom, --outpaint-left, --outpaint-right)')
# not implemented... yet:
#@click.option('--outpaint-fade-to', default="black", help='The color to fade to for outpainting.')
#@click.option('--outpaint-fade-amount', default=0.0, help='The amount to fade to the fade-to color at the edge of the outpainted area.')
#@click.option('--outpaint-fill-mode', default="repeat", help='The fill mode to use for outpainting.')
@click.option('--outpaint-prompt', default=None, help='The prompt to use for outpainting. Can be overridden per side.')
@click.option('--outpaint-negative-prompt', default=None, help='The negative prompt to use for outpainting the image. Can be overridden per side.')
@click.option('--outpaint-guidance-scale', default=9.0, help='The guidance scale to use for processing.')
@click.option('--outpaint-num-inference-steps', default=50, help='The number of steps to use for processing.')
@click.option('--outpaint-feather', default=20, help='The width of the blended transition between the outpainted area and the original images edge pixels ')
@click.option('--outpaint-top', default=0, help='How many pixels to add to the top of the image with outpainting.')
@click.option('--outpaint-top-prompt', default=None, help='The prompt to use for outpainting the top side.')
@click.option('--outpaint-top-negative-prompt', default=None, help='The negative prompt to use for outpainting the top side.')
@click.option('--outpaint-bottom', default=0, help='How many pixels to add to the bottom of the image with outpainting.')
@click.option('--outpaint-bottom-prompt', default=None, help='The prompt to use for outpainting the bottom side.')
@click.option('--outpaint-bottom-negative-prompt', default=None, help='The negative prompt to use for outpainting the bottom side.')
@click.option('--outpaint-left', default=0, help='How many pixels to add to the left of the image with outpainting.')
@click.option('--outpaint-left-prompt', default=None, help='The prompt to use for outpainting the left side.')
@click.option('--outpaint-left-negative-prompt', default=None, help='The negative prompt to use for outpainting the left side.')
@click.option('--outpaint-right', default=0, help='How many pixels to add to the right of the image with outpainting.')
@click.option('--outpaint-right-prompt', default=None, help='The prompt to use for outpainting the right side.')
@click.option('--outpaint-right-negative-prompt', default=None, help='The negative prompt to use for outpainting the right side.')
#@click.group(name="Upscaling ")
@click.option('--upscale', default=False, is_flag=True, help='Whether to upscale the image. (default: False)')
@click.option('--upscale-prompt', default=None, help='The prompt to use for guidance when upscaling the image.')
@click.option('--upscale-negative-prompt', default=None, help='The negative prompt to use for the image.')
@click.option('--upscale-tile-size', default=128, help='The size of the tiles to use for upscaling.')
@click.option('--upscale-tile-overlap', default=48, help='The overlap of the tiles to use for upscaling.')
@click.option('--upscale-guidance-scale', default=9.0, help='The guidance scale to use for upscaling.')
@click.option('--upscale-num-inference-steps', default=10, help='The number of steps to use for upscaling.')
@click.option('--upscale-add-predicted-noise', default=None, type=float, help='The noise level to use for upscaling.')
@click.option('--upscale-noise-level', default=None, type=float, help='The noise level to use for upscaling.')
#@click.option('--eta', default=None, help='The eta parameter to use for processing.')
def main(
			source_images, dest_dir, dest_name, 
	 		prompt, negative_prompt,
	 		outpaint_aspect_ratio, outpaint_fill_only, outpaint_prompt, outpaint_negative_prompt, 
      outpaint_guidance_scale, outpaint_num_inference_steps, outpaint_feather,
      outpaint_top, outpaint_top_prompt, outpaint_top_negative_prompt, 
      outpaint_bottom, outpaint_bottom_prompt, outpaint_bottom_negative_prompt, 
      outpaint_left, outpaint_left_prompt, outpaint_left_negative_prompt, 
      outpaint_right, outpaint_right_prompt, outpaint_right_negative_prompt, 
      upscale, upscale_prompt, upscale_negative_prompt, upscale_tile_size, upscale_tile_overlap, 
      upscale_guidance_scale, upscale_num_inference_steps, upscale_add_predicted_noise, upscale_noise_level
    ):
	
	# loop over the source images
	for source_image in source_images:
		
		# print out the source image
		print(f"Processing {source_image}...")

		# load the source image
		srcimg = Image.open(str(pathlib.Path(source_image).expanduser().resolve()))
		
		# figure out the destination path based on dest_dir and dest_name options (or defaults)
		if dest_name == None:
			destname = pathlib.Path(source_image).stem + ".out" + pathlib.Path(source_image).suffix
		else:
			destname = dest_name.format(name=pathlib.Path(source_image).stem, ext=pathlib.Path(source_image).suffix)
		if dest_dir == None:
			destpath = pathlib.Path(source_image).expanduser().resolve().parent / destname
		else:
			destpath = pathlib.Path(dest_dir).expanduser().resolve() / destname
		dest_image = str(destpath)

		# figure out the requested image's dimensions
		srcw = srcimg.width
		srch = srcimg.height
		srcar = srcw / srch
		if outpaint_aspect_ratio != None:
			if len(outpaint_aspect_ratio.split(":")) == 2:
				arw, arh = outpaint_aspect_ratio.split(":")
				ar = float(arw) / float(arh)
			else:
				ar = float(outpaint_aspect_ratio)
			if ar > srcar:
				# The aspect ratio is wider than the source image.
				# We need to add pixels to the left and right.
				outpaint_left = int((ar * srch - srcw) / 2)
				outpaint_right = int((ar * srch - srcw) / 2)
				outpaint_top = 0
				outpaint_bottom = 0
			elif ar < srcar:
				# The aspect ratio is taller than the source image.
				# We need to add pixels to the top and bottom.
				outpaint_top = int((srcw / ar - srch) / 2)
				outpaint_bottom = int((srcw / ar - srch) / 2)
				outpaint_left = 0
				outpaint_right = 0

		# Set default prompts if specific prompts are not set.
		if prompt == None:
			prompt = ""
		if outpaint_prompt == None:
			outpaint_prompt = prompt
		if outpaint_negative_prompt == None:
			outpaint_negative_prompt = negative_prompt
		if upscale_prompt == None:
			upscale_prompt = prompt
		if upscale_negative_prompt == None:
			upscale_negative_prompt = negative_prompt

		def updatefunc(im, rect):
				if rect != None:
					outim = im.copy()
					outim.resize((srcw/2, srch/2), resample=Image.LANCZOS)
					rl, rt, rr, rb = rect
					import PIL.ImageDraw
					draw = PIL.ImageDraw.Draw(outim)
					draw.rectangle((rl/8, rt/8, rr/8, rb/8), outline="red", width=1)
					outim.save(str(pathlib.Path(dest_image).expanduser().resolve()))

		im = srcimg
		if outpaint_top > 0 or outpaint_bottom > 0:
			print(f"Outpainting top {outpaint_top}px and bottom {outpaint_bottom}px ...")
			im = outpaint_vertical(im, outpaint_top, outpaint_bottom, feather_pixels_top=outpaint_feather, feather_pixels_bottom=outpaint_feather,
			  options=PipelineOptions(prompt=outpaint_prompt, negative_prompt=outpaint_negative_prompt, 
			       guidance_scale=outpaint_guidance_scale, num_inference_steps=outpaint_num_inference_steps),
			  top_options=PipelineOptions(prompt=outpaint_top_prompt, negative_prompt=outpaint_top_negative_prompt),
			  bottom_options=PipelineOptions(prompt=outpaint_bottom_prompt, negative_prompt=outpaint_bottom_negative_prompt),
			  fill_only=outpaint_fill_only
			)

		if outpaint_left > 0 or outpaint_right > 0:
			print(f"Outpainting left {outpaint_left}px and right {outpaint_right}px ...")
			im = outpaint_horizontal(im, outpaint_left, outpaint_right, feather_pixels_left=outpaint_feather, feather_pixels_right=outpaint_feather,
			  options=PipelineOptions(prompt=outpaint_prompt, negative_prompt=outpaint_negative_prompt, 
			       guidance_scale=outpaint_guidance_scale, num_inference_steps=outpaint_num_inference_steps),
			  left_options=PipelineOptions(prompt=outpaint_left_prompt, negative_prompt=outpaint_left_negative_prompt),
			  right_options=PipelineOptions(prompt=outpaint_right_prompt, negative_prompt=outpaint_right_negative_prompt),
			  fill_only=outpaint_fill_only
			)

		if upscale:
			print(f"Upscaling ...")
			im = upscale_image(source_image=im, tile_size=upscale_tile_size, tile_overlap=upscale_tile_overlap, pipeline_options=PipelineOptions(prompt=upscale_prompt, negative_prompt=upscale_negative_prompt, guidance_scale=upscale_guidance_scale, num_inference_steps=upscale_num_inference_steps, noise_level=upscale_noise_level, add_predicted_noise=upscale_add_predicted_noise), updatefunc=None)

		print(f"Saving {im.width}x{im.height} output to: {dest_image}")
		im.save(str(pathlib.Path(dest_image).expanduser().resolve()))

		print(f"Done processing {source_image}.")

if __name__ == '__main__':
		main()

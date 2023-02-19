# sd2upscaler

A tiling prompt-guided super-resolution and outpainting CLI tool based on [huggingface](huggingface.co) [diffusers](https://github.com/huggingface/diffusers) library and [stability.ai](https://stability.ai)'s [Stable Diffusion v2.1](https://github.com/stabilityai/stable-diffusion-v2) upscaling and outpainting models.

---

```
Usage: sd2upscaler.py [OPTIONS] [SOURCE_IMAGES]...

Options:
  --dest-dir TEXT                 The directory to save the output images to.
                                  If not specified, the output images will be
                                  saved to the same directory as the source
                                  image.
  --dest-name TEXT                Name template for output images.
                                  Substiutions: {parent} parent dir of source
                                  image, {name} name of source image, {ext}
                                  extension of source image. Default:
                                  {name}-sd2out{ext}
  --prompt TEXT                   The default prompt to use.
  --negative-prompt TEXT          The default negative prompt to use.
  --outpaint-aspect-ratio TEXT    Outpaint the image to the specified aspect
                                  ratio (overrides --outpaint-top, --outpaint-
                                  bottom, --outpaint-left, --outpaint-right)
  --outpaint-fill-only            Output the pre-processed image normally fed
                                  through the SD2 inpainting model as-is.
                                  (overrides --outpaint-top, --outpaint-
                                  bottom, --outpaint-left, --outpaint-right)
  --outpaint-prompt TEXT          The prompt to use for outpainting. Can be
                                  overridden per side.
  --outpaint-negative-prompt TEXT
                                  The negative prompt to use for outpainting
                                  the image. Can be overridden per side.
  --outpaint-guidance-scale FLOAT
                                  The guidance scale to use for processing.
  --outpaint-num-inference-steps INTEGER
                                  The number of steps to use for processing.
  --outpaint-feather INTEGER      The width of the blended transition between
                                  the outpainted area and the original images
                                  edge pixels
  --outpaint-top INTEGER          How many pixels to add to the top of the
                                  image with outpainting.
  --outpaint-top-prompt TEXT      The prompt to use for outpainting the top
                                  side.
  --outpaint-top-negative-prompt TEXT
                                  The negative prompt to use for outpainting
                                  the top side.
  --outpaint-bottom INTEGER       How many pixels to add to the bottom of the
                                  image with outpainting.
  --outpaint-bottom-prompt TEXT   The prompt to use for outpainting the bottom
                                  side.
  --outpaint-bottom-negative-prompt TEXT
                                  The negative prompt to use for outpainting
                                  the bottom side.
  --outpaint-left INTEGER         How many pixels to add to the left of the
                                  image with outpainting.
  --outpaint-left-prompt TEXT     The prompt to use for outpainting the left
                                  side.
  --outpaint-left-negative-prompt TEXT
                                  The negative prompt to use for outpainting
                                  the left side.
  --outpaint-right INTEGER        How many pixels to add to the right of the
                                  image with outpainting.
  --outpaint-right-prompt TEXT    The prompt to use for outpainting the right
                                  side.
  --outpaint-right-negative-prompt TEXT
                                  The negative prompt to use for outpainting
                                  the right side.
  --upscale                       Whether to upscale the image. (default:
                                  False)
  --upscale-prompt TEXT           The prompt to use for guidance when
                                  upscaling the image.
  --upscale-negative-prompt TEXT  The negative prompt to use for the image.
  --upscale-tile-size INTEGER     The size of the tiles to use for upscaling.
  --upscale-tile-overlap INTEGER  The overlap of the tiles to use for
                                  upscaling.
  --upscale-guidance-scale FLOAT  The guidance scale to use for upscaling.
  --upscale-num-inference-steps INTEGER
                                  The number of steps to use for upscaling.
  --upscale-add-predicted-noise FLOAT
                                  The noise level to use for upscaling.
  --upscale-noise-level FLOAT     The noise level to use for upscaling.
  --help                          Show this message and exit.
```


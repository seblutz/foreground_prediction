# Foreground color prediction through inverse compositing
This repository contains the inference code for our WACV 2021 paper [Foreground color prediction through inverse compositing](https://v-sense.scss.tcd.ie/wp-content/uploads/2020/11/WACV_submission_IEEE.pdf) ([Supplementary](https://v-sense.scss.tcd.ie/wp-content/uploads/2020/11/WACV_supplementary.pdf)).

## Requirements
**Packages**
* torch (Tested on 1.7)
* numpy
* opencv-python
* Pillow
* tqdm

## Run the code
First, you will need to download the trained weights and save them in the _weights_ folder. See [here](https://github.com/seblutz/foreground_prediction/tree/main/weights) for download links.  
To run the code, use  
```
python run.py <input directory> <output directory>
```  
The `<input directory>` needs to contain the subdirectories _rgb_ and _trimap_ with matching image names. The `<output directory>` is the path in which the results will be saved. To run the code on a set of example images from the [alphamatting.com](http://alphamatting.com/) benchmark, run:  
```
python run.py example/input example/output
```

There are a few optional parameters.
* `-t <X>` will do `X` iterations of the recurrent inference machine. The default is `5`.
* `-w <path>` will search for the network weights in the `<path>` directory.
* `-workers <N>` will set the number of workers for data loading to `N`. The default is `4`.
* `-tile_size <S>` will set the spatial size of the tile that should be processed from the image. Instead of the whole image, `S x S` tiles will be processed. To process the image without tiling, set the tile size to larger than the image. The default is `512`.
* `-all` will save all itermediate results in the recurrent inference machine.
* `-cpu` will process the image on the CPU.
* `-alpha <path>` will use the alphas from the `<path>` directory instead of using GCA-Matting.

## Citation
If you find this code useful for your work or research, please consider citing:
```
@inproceedings{Lutz2021foreground,
  title = {Foreground color prediction through inverse compositing},
  author = {Sebastian Lutz and Aljosa Smolic},
  url = {https://v-sense.scss.tcd.ie/wp-content/uploads/2020/11/WACV_submission_IEEE.pdf},
  year = {2021},
  date = {2021-01-09},
  booktitle = {Winter Conference on Applications of Computer Vision 2021 (WACV 2021)}
}
```

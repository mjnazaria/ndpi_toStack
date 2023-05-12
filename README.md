
This repository contains a python GUI that can be used to convert ndpi/tiff images of brain slices to an aligned stack.
It uses [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything/tree/main) for segmenting brain sections 
and have functionalities for removing mislabled objects and combining brain sections that are ripped apart. After cloning the repo, one of 
the SAM's model checkpoints should be downloaded from [here](https://github.com/facebookresearch/segment-anything/tree/main) and put in the main directory.

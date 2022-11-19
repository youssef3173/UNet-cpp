# Semantic Segmentation: Unet Trained on Cityscapes Dataset
The Unet model used for image segmentation, ...

## Dataset:
The Cityscapes available on [a link](https://www.cityscapes-dataset.com/) is a dataset for research purposes only. This dataset contains images from stereo video sequences recorded in street scenes in 50 different cities, as well as the respective annotation of each image.
 
## Requirements:
- python3.8+
- torch
- torchvision
- matplotlib
- opencv
- tqdm


## Train the Unet:

```
python3 main.py
```


## Build and compile the Project:
```
bash shell.sh
cd build
make
```

## Apply the Model to an image: 
```
./UNet ../Unet-TSM.pt ../test_images/sample-0.png ../RESULTS.png
```

# Semantic Segmentation: Unet Trained on Cityscapes Dataset
The Unet model used for image segmentation, ...


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

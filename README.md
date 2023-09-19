# Evaluation Metrics

We released Evaluation Metrics for commmently used image metrics code to help test performance.  
in this work, we released the following metrics for testing:
1. SSIM
2. PSNR
3. FID
4. Evaluation Metrics of image clarity:
  1. SMD
  2. SMD2
  3. Brenner
  4. Laplacian
  5. Variance
  6. Energy
  7. Vollath
  8. Entropy


Illustrate:
* SSIM: (structural similarity index measure), the larger the value of the structural similarity index.  
  The higher the similarity between the two signals.

* PSNR: (Peak Signal to Noise Ratio), used to measure the difference between two images, such as a compressed image and an original image, to evaluate the quality of a compressed image; a restored image and ground truth, to evaluate the performance of a restoration algorithm, etc.  
  The larger the PSNR, the smaller the difference between the two images.

* FID: represents the diversity and quality of the generated images.  
  The better the image diversity, the better the quality.

* Image clarity  
  (1) SMD  
  (2) SMD2  
  (3) Brenner  
  (4) Laplacian  
  (5) Variance  
  (6) Energy  
  (7) Vollath  
  (8) Entropy  
  Those metrics all represent image sharpness evaluation methods.  
  The larger the value, the higher the clarity.

### Recommondation of our works
This repo is maintaining by authors, if you have any questions, please contact us at issue tracker.

**The official repository with Pytorch**

## Requirements
* [python](https://www.python.org/download/releases/)（We use version 3.7)

## Metrics testing
if you want to test the metrics of SSIM, PSNR and FID, you can run the following:
bash
``` 
python evaluate.py --img_folder1 {} --img_folder2 {}
```


if you want to test the metrics of SMD, SMD2, Brenner, Laplacian, Variance, Energy, Vollath and Entropy, you can run the following:
bash
``` 
python evaluate_clarity.py --img_folder {}
```


The directories of img_folder, img_folder1 and img_folder2 are arranged like this
```
img_folder
├── 0, 1, ... 28.jpg（or {'bmp', 'jpg', 'jpeg', 'pgm', 'png'}）

```

## Star
Please star this project if you use this repository in your research. Thank you!
# celebA-HQ-dataset-download
While working with celebA-HQ dataset I found it quite difficult to generate the dataset, so I collected the following scripts and dockerized it to make life a little bit easier.  

To get the celebA-HQ dataset, you need to  
 a) download the celebA dataset `download_celebA.py`,  
 b) unzip celebA files with `p7zip`,  
 c) move `Anno` files to `celebA` folder,  
 d) download some extra files, `download_celebA_HQ.py`,  
 e) do some processing to get the HQ images `make_HQ_images.py`.

The size of the final dataset is 89G. However, you will need a bit more storage to be able to run the scripts.

# Usage
## Docker

If you have Docker installed, run the following command from the root directory of this project:

`docker build -t celeba-hq . && docker run -it -v $(pwd):/data celebahq`

By default, this will create the dataset in same directory. To put it elsewhere, replace `$(pwd)` with the absolute path to the desired output directory.

### Prebuilt Docker Image
I also have a pre-built docker image at `suvojit0x55aa/celeba-hq`. You can just docker run without cloning the repo even ! 
```
docker run -it -v $(pwd):/data suvojit0x55aa/celeba-hq
```
## Running it locally
1) Clone the repository
```
git clone https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download.git
cd celebA-HQ-dataset-download
```

2) Install necessary packages (Because specific versions are required Conda is recomended)
 * Install miniconda https://conda.io/miniconda.html
 * Create a new environement
 ```
 conda create -n celebaHQ python=3
 source activate celebaHQ
 ```
 * Install the packages
 ```
 conda install jpeg=8d tqdm requests pillow==3.1.1 urllib3 numpy cryptography scipy
 pip install opencv-python==3.4.0.12 cryptography==2.1.4
 ```
 * Install 7zip (On Ubuntu)
 ```
 sudo apt-get install p7zip-full
 ```

3) Run the scripts
```
./create_celebA-HQ.sh <dir_to_save_files>
```
where `<dir_to_save_files>` is the directory where you wish the data to be saved.

4) Go watch a movie, theses scripts will take a few hours to run depending on your internet connection and your CPU power. By default the script will launch as many jobs as you have cores on your CPU. If you want to change this behaviour change the `create_celebA-HQ.sh` script. The final HQ images will be saved as `.jpg` files in the `<dir_to_save_files>/celeba-hq` folder.

# Pre-Calculated Dataset
This script generated the dateset with original names from CelebA. If you're okay with a version of the dataset that is named index wise you can save a lot of time and effort and download it from this convenient [Google Drive link](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P?usp=sharing).
# Remark
This script has lot of specific dependencies and is likely to break somewhere, but if it executes until the end, you should obtain the correct dataset. However the docker is pretty fool-proof, so do use it if you can.

# Sources
This code is inspired by these files
* https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
* https://github.com/andersbll/deeppy/blob/master/deeppy/dataset/celeba.py
* https://github.com/andersbll/deeppy/blob/master/deeppy/dataset/util.py
* https://github.com/nperraud/download-celebA-HQ
* https://github.com/willylulu/celeba-hq-modified

# Citing the dataset
You probably want to cite the paper "Progressive Growing of GANs for Improved Quality, Stability, and Variation" that was submitted to ICLR 2018 by Tero Karras (NVIDIA), Timo Aila (NVIDIA), Samuli Laine (NVIDIA), Jaakko Lehtinen (NVIDIA and Aalto University).

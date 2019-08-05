WORKDIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $WORKDIR

# Download CelebA unaligned images
python $WORKDIR/download_celebA.py $1

# Download CelebA-HQ deltas
python $WORKDIR/download_celebA_HQ.py $1

# Extract CelebA images
cat $1/celebA/img_celeba.7z.0* > $1/celebA/img_celeba.7z
7z x $1/celebA/img_celeba.7z -o$1/celebA

# Create CelebA-HQ images
python $WORKDIR/create_celeba_HQ.py \
        --h5_filename celeba-hq-1024x1024.h5 \
        --celeba_dir $1/celebA \
        --delta_dir $1/celebA-HQ \
        --output_dir $1
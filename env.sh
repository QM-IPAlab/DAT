source /opt/conda/etc/profile.d/conda.sh
source ~/.bashrc
conda env create -f environment.yml
conda activate dat
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install loguru

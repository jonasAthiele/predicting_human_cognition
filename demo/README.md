# Demo

## Set up

  1)	Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html.
  
  2)	Use command prompt (e.g., Anaconda prompt) to create a new environment with requirements:
  	
          	conda create -n  "myenv" python=3.8.18 pip
          	conda activate myenv
          	conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch
          	conda install captum==0.5.0 -c pytorch
          	pip install mne==1.3.1
          	pip install -U mne-connectivity==0.5.0
          	pip install -U scikit-learn==1.3.2
          	pip install -U matplotlib==3.5.2
          	pip install seaborn==0.11.2
          	pip install pillow==9.0.0

Note: If you want to use Cuda (run torch models on GPU) please refer to: https://pytorch.org/get-started/locally/ for details on how to install.



## Run demo



## Visualize results of manuscript

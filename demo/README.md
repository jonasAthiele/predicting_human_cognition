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

`demo_run`: The demo runs minimal examples of intelligence prediction within the main sample (610 subjects, 5-fold cross-validation). Note that only one iteration of model training is performed (either with real or permuted intelligence scores) and that the output is the result of this specific iteration. For results in the manuscript, 10 iterations with correctly assigned scores and 100 permutations with permuted scores were performed. Note also that confounding variables (such as age and handedness) are not correctly assigned as those variables require permitted access to restricted data of the Human Connectome Project.

Run with command prompt:

python run_demo.py --state 'state' --score 'score' --option 'option' --link_selection 'link_selection'

&nbsp;&nbsp;&nbsp;Parameters:

  &nbsp;&nbsp;&nbsp;&nbsp;'state': rest, WM, gambling, motor, language, social, realtional, emotion, latent_states, latent_task\
  &nbsp;&nbsp;&nbsp;&nbsp;'score': g_score, gf_score, gc_score\
  &nbsp;&nbsp;&nbsp;&nbsp;'option': real, permutation (real = correctly assigned scores, permutation = permuted scores, only for link_selections: all, within_between, allbutone, one)\
  &nbsp;&nbsp;&nbsp;&nbsp;'link_selection': all, within_between, allbutone, one, nodes_20, links_40, pfc_extended, pfit, md_diachek (default), md_duncan,  cole 


Note that link_selection "nodes_n" selects links between n random nodes, "links_n" selectes nChoosek(n,2) random links.\
Note that "pfc_extended", "pfit", "md_diachek", "md_duncan", "cole" selects links corresponding to the respective intelligence theory.\
Note that the computation time depends on the link_selection (~5-30 minutes)


Examples:
      
            python run_demo.py --state language --score g_score --option real --link_selection all
            python run_demo.py --state WM --score gf_score --option real --link_selection within_between
            python run_demo.py --state WM --score g_score --link_selection nodes_20
            python run_demo.py --state WM --score g_score --link_selection links_10
            python run_demo.py --state WM --score g_score --link_selection md_diachek

## Visualize results of manuscript

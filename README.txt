Codigo para rodar o colab 
(https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/mpnn/examples/proteinmpnn_in_jax.ipynb)
diretamente na maquina
Requisitos:
Criar ambiente virtual python
instalar as bibliotecas "pip install colabdesign matplotlib pandas jax tqdm"
Baixar os arquivos params do AlphaFold em https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar.
Extrai-los numa pasta 'params'
Criar uma pasta output/all_pdb
Executar o comando pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
Para liberar o uso de GPU
Adicionar o .pdb e .txt na pasta


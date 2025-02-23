import os
import re
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from tkinter import Tk, filedialog
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model

# Configuração de avisos
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração do progresso
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

# Criação de diretórios
os.makedirs("output", exist_ok=True)

def upload_files():
    """Função para seleção de arquivos PDB e TXT."""
    Tk().withdraw()  # Oculta a janela principal do Tkinter
    print("Selecione os arquivos PDB.")
    pdb_filepaths = filedialog.askopenfilenames(filetypes=[("PDB files", "*.pdb")])
    print("Selecione os arquivos TXT correspondentes.")
    txt_filepaths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])

    pdb_files = {os.path.basename(fp): open(fp).read() for fp in pdb_filepaths}
    txt_files = {os.path.basename(fp): open(fp).read() for fp in txt_filepaths}

    if len(pdb_files) != len(txt_files):
        raise ValueError("O número de arquivos PDB e TXT não é igual!")

    print(f"Carregados {len(pdb_files)} arquivos PDB e {len(txt_files)} arquivos TXT.")
    return pdb_files, txt_files

def process_files(pdb_content, txt_content, pdb_filename, results, num_seqs, sampling_temp, homooligomer):
    model_name = "v_48_020"
    chains = "B"
    fix_pos = txt_content

    # Preparar modelo MPNN
    mpnn_model = mk_mpnn_model(model_name)
    mpnn_model.prep_inputs(pdb_filename=pdb_content, chain=chains, homooligomer=homooligomer,
                           fix_pos=fix_pos, verbose=True)
    out = mpnn_model.sample(num=num_seqs//32, batch=32, temperature=sampling_temp, rescore=homooligomer)

    for n in range(num_seqs):
        results.append({
            "pdb_file": pdb_filename,
            "score": out["score"][n],
            "seqid": out["seqid"][n],
            "seq": out["seq"][n]
        })

    return chains, out

def runAlphaFold(pdb_path, num_seqs, sampling_temp, homooligomer, chains, out):
    num_models = 1
    num_recycles = 3
    use_multimer = False
    use_templates = False
    arq_name = pdb_path.replace(".pdb", "")

    af_model = mk_af_model(use_multimer=use_multimer, use_templates=use_templates, best_metric="dgram_cce")
    af_model.prep_inputs(pdb_path, chains, homooligomer=homooligomer)

    with tqdm.tqdm(total=out["S"].shape[0], bar_format=TQDM_BAR_FORMAT) as pbar:
        for n, S in enumerate(out["S"]):
            seq = S[:af_model._len].argmax(-1)
            af_model.predict(seq=seq, num_recycles=num_recycles, num_models=num_models, verbose=False)
            af_model.save_current_pdb(f"output/all_pdb/{arq_name}_n{n}.pdb")
            pbar.update(1)

    af_model.save_pdb(f"output/best.pdb")

    data = []
    labels = ["dgram_cce", "plddt", "ptm", "i_ptm", "rmsd", "composite", "mpnn", "seqid", "seq"]
    for n, af in enumerate(af_model._tmp["log"]):
        data.append([af["dgram_cce"], af["plddt"], af["ptm"], af["i_ptm"], af["rmsd"], af["composite"],
                     out["score"][n], out["seqid"][n], out["seq"][n]])

    df = pd.DataFrame(data, columns=labels)
    df.to_csv('output/alphafold_results.csv')

def main():
    num_seqs = 32
    sampling_temp = 0.1
    homooligomer = False

    pdb_data, txt_data = upload_files()
    results = []

    for pdb_filename, pdb_content in pdb_data.items():
        txt_filename = pdb_filename.replace(".pdb", ".txt")
        if txt_filename in txt_data:
            txt_content = txt_data[txt_filename]
            chains, out = process_files(pdb_content, txt_content, pdb_filename, results, num_seqs, sampling_temp, homooligomer)
            runAlphaFold(pdb_filename, num_seqs, sampling_temp, homooligomer, chains, out)
        else:
            print(f"Arquivo .txt correspondente para {pdb_filename} não encontrado.")

    # Criar DataFrame com os resultados
    df = pd.DataFrame(results, columns=["pdb_file", "score", "seqid", "seq"])
    df.to_csv('output/mpnn_results_final.csv')
    print("Resultados salvos em output/mpnn_results_final.csv")

# Executar o código
if __name__ == "__main__":
    main()

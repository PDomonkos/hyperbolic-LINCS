from cmapPy.pandasGEXpress.parse import parse
import pandas as pd
import numpy as np 


def load_data(L1000FWD_path, SigComLINCS_path):

    GCToo_landmark = parse(L1000FWD_path + "CD_signatures_LM_42809x978.gctx")
    GCToo_full = parse(L1000FWD_path + "CD_signatures_full_42809x22268.gctx")

    df_signitures = pd.read_csv(L1000FWD_path + "Signature_Graph_CD_center_LM_sig-only_16848nodes.csv")
    L1000FWD_signature_indices = df_signitures.sig_id.values

    data = {
        "L1000FWD": {
            "full": {
                "signatures": GCToo_full.data_df.loc[L1000FWD_signature_indices, :].values,
                "metadata": df_signitures
            },
            "landmark": {
                "signatures": GCToo_landmark.data_df.loc[L1000FWD_signature_indices, :].values,
                "metadata": df_signitures
            },
        }
    }

    GCToo_chemical = parse(SigComLINCS_path + "cp_coeff_mat.gctx")
    df_chemical_signatures = GCToo_chemical.data_df.T
    df_chemical_meta = GCToo_chemical.col_metadata_df

    # map moa
    df_mol_meta = pd.read_csv(SigComLINCS_path + "LINCS_small_molecules.tsv", sep='\t', header=0)
    new_moa = []
    for v in df_mol_meta.moa.values:
        if "," in v:
            new_moa += ["unknown"]
        elif "-" == v:
            new_moa += ["unknown"]
        elif "nan" == v:
            new_moa += ["unknown"]
        else:
            new_moa += [v]
    df_mol_meta.moa = np.array(new_moa)

    data["SigComLINCS"] = {}
    for cell_line in ["NEU" ,"NPC", "HELA", "A375"]:
        df_cell_line_chemical_meta = df_chemical_meta[df_chemical_meta["cell_line"] == cell_line].reset_index()
        df_cell_line_chemical_meta = df_cell_line_chemical_meta.merge(df_mol_meta[["pert_name", "moa"]], how = "left", on = "pert_name")
        df_cell_line_chemical_meta = df_cell_line_chemical_meta.drop_duplicates("cid").set_index("cid")
        df_cell_line_chemical_meta = df_cell_line_chemical_meta.rename(columns={"moa": "MOA", "pert_time": "Time", "pert_dose": "Dose","pert_name": "Perturbation","cell_line": "Cell"})
        df_cell_line_chemical_signatures = df_chemical_signatures.loc[df_cell_line_chemical_meta.index.values].values

        data["SigComLINCS"][cell_line] = {
            "signatures": df_cell_line_chemical_signatures,
            "metadata": df_cell_line_chemical_meta
        }

    return data
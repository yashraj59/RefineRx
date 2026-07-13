"""load_arc_state.py — construct real StateTransitionPerturbationModel + load ARC pretrained weights.
Works for both ST-HVG and ST-SE Replogle checkpoints (SE adds a gene_decoder: embed-space -> gene-space)."""
import sys, glob, torch, pickle, yaml
sys.path.insert(0, "/workspace/halt/state_repo/src")

def load_arc(cache_glob, device="cuda", cell_line=None, split=None):
    # scope globs to a specific split/cell_line dir when given (the multi-line cache holds all 8);
    # prefer final.ckpt, fall back to best.ckpt (minimal downloads may omit final).
    sub = ""
    if split and cell_line: sub = f"/**/{split}/{cell_line}"
    def _find(pat):
        hits = glob.glob(f"{cache_glob}{sub}/**/{pat}", recursive=True) if sub else glob.glob(f"{cache_glob}/**/{pat}", recursive=True)
        return hits
    ckpt_hits = _find("checkpoints/final.ckpt") or _find("checkpoints/best.ckpt")
    ckpt = ckpt_hits[0]
    cfg  = _find("config.yaml")[0]
    vardims = _find("var_dims.pkl")[0]
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = sd["hyper_parameters"]; state = sd["state_dict"]
    vd = pickle.load(open(vardims, "rb"))
    d = yaml.safe_load(open(cfg)); mk = d["model"]["kwargs"]
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    init = dict(
        input_dim=vd["input_dim"], hidden_dim=hp["hidden_dim"], output_dim=vd["output_dim"],
        pert_dim=vd["pert_dim"], batch_dim=hp.get("batch_dim"), gene_dim=vd.get("gene_dim"),
        transformer_backbone_key=mk["transformer_backbone_key"],
        transformer_backbone_kwargs=dict(mk["transformer_backbone_kwargs"]),
        output_space=mk.get("output_space","gene"),
        predict_residual=mk.get("predict_residual",True),
        distributional_loss=mk.get("distributional_loss","energy"),
    )
    for k,v in mk.items():
        if k not in init and k != "transformer_backbone_kwargs": init.setdefault(k,v)
    for k in ("embed_key","control_pert","gene_names","decoder_cfg","dropout","cell_set_len","hvg_dim"):
        if k in hp: init.setdefault(k, hp[k])
    model = StateTransitionPerturbationModel(**init)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, vd, hp, dict(missing=list(missing), unexpected=list(unexpected))

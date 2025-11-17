# ================================================================
# Inférence 3D pour IRM multi-modales avec U-Net 3D (PyTorch)
# Entrée  : (4, Z, Y, X)  -> [t1_pre, t1_gd, flair]
# Sortie  : NIfTI binaire (Z, Y, X)
# Auteurs : Marine Camus
# ================================================================

import os
from pathlib import Path
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------- CONFIG -------------

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/Users/marine/Desktop/BrainMetShare/poids/3D_U-Net_BraTS_ckpt.tar" # chemin vers les poids
TEST_ROOT = "/Users/marine/Desktop/BrainMetShare/train" # chemin vers la data
OUT_DIR   = "./runs/infer_unet3d"
NORMALIZE = "zscore"
THRESH    = 0.5

ROI_SIZE  = (64, 128, 128)
OVERLAP   = 0.5


# ------------- I/O et NORMALISATION -------------
def load_nii(path):
    """Charge NIfTI -> (Z,Y,X) float32 et retourne aussi l'objet img pour l'affine/header."""
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)    # (X,Y,Z)
    arr = np.transpose(arr, (2,1,0))         # -> (Z,Y,X)
    return arr, img

def zscore(x, eps=1e-6):
    m, s = x.mean(), x.std()
    if s < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (s + eps)

def minmax01(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def normalize_vol(vol, mode):
    if mode == "zscore":
        return zscore(vol)
    if mode == "minmax":
        return minmax01(vol)
    return vol.astype(np.float32)

# ------------- DATASET (test) -------------
class BrainMetsDataset3DTest(Dataset):
    """
    Dataset 3D multi-modal (test/inférence).
    Cherche [t1_pre, t1_gd, flair, bravo]. Si 'bravo' absent (version dossiers), duplique t1_pre.
    """
    def __init__(self, root_dir, normalize="zscore"):
        self.root_dir = Path(root_dir)
        self.normalize = normalize
        self.samples = []

        case_dirs = sorted([p for p in self.root_dir.glob("*") if p.is_dir()])
        for base in case_dirs:
            named = {
                "t1_pre":  list(base.glob("t1_pre*.nii*")),
                "t1_gd":   list(base.glob("t1_gd*.nii*")),
                "flair":   list(base.glob("flair*.nii*")),
                "bravo":   list(base.glob("bravo*.nii*")),
            }
            ok_named = all(len(named[k])>=1 for k in ["t1_pre","t1_gd","flair"]) and (len(named["bravo"])>=1)
            num_ok = all((base/str(s)).exists() for s in ["0","1","3"])

            if ok_named:
                self.samples.append({
                    "case_id": base.name,
                    "paths": [named["t1_pre"][0], named["t1_gd"][0], named["flair"][0], named["bravo"][0]],
                })
            elif num_ok:
                p0 = sorted((base/"0").glob("*.nii*"))  # t1_gd
                p1 = sorted((base/"1").glob("*.nii*"))  # t1_pre
                p3 = sorted((base/"3").glob("*.nii*"))  # flair
                if not (p0 and p1 and p3):
                    continue
                bravo_guess = list(base.glob("bravo*.nii*"))
                p_bravo = bravo_guess[0] if bravo_guess else p1[0]
                self.samples.append({
                    "case_id": base.name,
                    "paths": [p1[0], p0[0], p3[0], p_bravo],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        it = self.samples[idx]
        vols = []
        ref_img = None
        for i, p in enumerate(it["paths"]):
            v, img = load_nii(p)
            if i == 0:
                ref_img = img
            v = normalize_vol(v, NORMALIZE)
            vols.append(v)
        img4 = np.stack(vols, axis=0).astype(np.float32)  # (4,Z,Y,X)
        return {"image": torch.from_numpy(img4), "case_id": it["case_id"], "ref_img": ref_img}

# ------------- U-NET 3D -------------

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels), 
        nn.ReLU(inplace=True),
        nn.Dropout(0.1 if out_channels <= 32 else 0.2 if out_channels <= 128 else 0.3),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),  
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Contraction path
        self.conv1 = double_conv(in_channels=in_channels, out_channels=16)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = double_conv(in_channels=16, out_channels=32)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = double_conv(in_channels=32, out_channels=64)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = double_conv(in_channels=64, out_channels=128)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        
        self.conv5 = double_conv(in_channels=128, out_channels=256)

        # Expansive path
        self.upconv6 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6 = double_conv(in_channels=256, out_channels=128)

        self.upconv7 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7 = double_conv(in_channels=128, out_channels=64)

        self.upconv8 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8 = double_conv(in_channels=64, out_channels=32)

        self.upconv9 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9 = double_conv(in_channels=32, out_channels=16)

        self.out_conv = nn.Conv3d(in_channels=16, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4) 

        c5 = self.conv5(p4)

        # Expansive path
        u6 = self.upconv6(c5)  
        u6 = torch.cat([u6, c4], dim=1)  
        c6 = self.conv6(u6)

        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        outputs = self.out_conv(c9)

        return outputs



# ------------- Fenêtre Hann 3D (pondération) -------------
def hann1d(L):
    if L == 1: return np.ones((1,), dtype=np.float32)
    return 0.5 * (1 - np.cos(2*np.pi*np.arange(L)/(L-1)))

def make_hann3d(sz):
    d,h,w = sz
    wd = hann1d(d)[:,None,None]
    wh = hann1d(h)[None,:,None]
    ww = hann1d(w)[None,None,:]
    win = wd * wh * ww
    return win.astype(np.float32)

# ------------- Sliding window 3D inference -------------
@torch.no_grad()
def sliding_window_infer_3d(model, vol4, roi_size=(64,128,128), overlap=0.5, device="cpu", thr=0.5):
    """
    vol: torch.Tensor (C, Z, Y, X)  avec C = 3 ou 4
    """
    model.eval()
    C, Z, Y, X = vol4.shape
    Dz, Dy, Dx  = roi_size

    # Pas (en voxels) -> stride = taille * (1 - overlap)
    sz = max(1, int(Dz * (1 - overlap)))
    sy = max(1, int(Dy * (1 - overlap)))
    sx = max(1, int(Dx * (1 - overlap)))

    # Déterminer les origines de fenêtres (borne max incluse à la fin)
    def starts(L, k):
        out = list(range(0, max(1, L - k + 1), max(1, int(k*(1-overlap)))))
        if out[-1] != L - k:
            out.append(L - k if L >= k else 0)
        return out

    zs = starts(Z, Dz)
    ys = starts(Y, Dy)
    xs = starts(X, Dx)

    vol4 = vol4.unsqueeze(0).to(device)  # (1, C, Z, Y, X)

    # Accumulateur et fenêtre de pondération
    acc   = torch.zeros((1, 1, Z, Y, X), dtype=torch.float32, device=device)
    wsum  = torch.zeros_like(acc)

    win_np = make_hann3d((Dz, Dy, Dx))
    win = torch.from_numpy(win_np).to(device)[None, None]  # (1,1,Dz,Dy,Dx)

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                z1, y1, x1 = z0+Dz, y0+Dy, x0+Dx

                # Extraire patch (pads si besoin)
                patch = vol4[:,:, z0:z1, y0:y1, x0:x1]  # (1,C,D',H',W')
                pd, ph, pw = Dz - patch.shape[2], Dy - patch.shape[3], Dx - patch.shape[4]
                if pd>0 or ph>0 or pw>0:
                    patch = nn.functional.pad(patch, (0,pw, 0,ph, 0,pd), mode="constant", value=0)

                # Prédiction
                logits = model(patch)   # (1, C_out, Dz, Dy, Dx)

                if logits.shape[1] == 1:
                    # Cas binaire classique
                    probs = torch.sigmoid(logits)       # (1,1,Dz,Dy,Dx)
                else:
                    # Cas multi-classe (C_out = 4 pour BraTS)
                    # On fait un softmax puis on agrège les classes "tumeur"
                    class_probs = torch.softmax(logits, dim=1)   # (1,4,Dz,Dy,Dx)
                    # On considère que canal 0 = fond, 1/2/3 = tumeur -> on les somme
                    tumor_prob = class_probs[:, 1:, ...].sum(dim=1, keepdim=True)  # (1,1,Dz,Dy,Dx)
                    probs = tumor_prob


                # Ajouter pondéré
                acc[:,:, z0:z0+Dz, y0:y0+Dy, x0:x0+Dx]  += probs * win
                wsum[:,:, z0:z0+Dz, y0:y0+Dy, x0:x0+Dx] += win

    probs_full = acc / torch.clamp_min(wsum, 1e-8)
    mask = (probs_full >= thr).float()

    return mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)  # (Z,Y,X)

# ------------- Sauvegarde NIfTI -------------
def save_mask_nifti(mask_zyx, ref_img, out_path):
    arr_xyz = np.transpose(mask_zyx, (2,1,0))  # (X,Y,Z)
    nii = nib.Nifti1Image(arr_xyz.astype(np.uint8), affine=ref_img.affine, header=ref_img.header)
    nii.set_data_dtype(np.uint8)
    nib.save(nii, str(out_path))

# ------------- DICE et IOU -------------
def dice_coefficient(pred, target, eps=1e-6):
    """
    pred, target : np.array (Z,Y,X) binaires {0,1}
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    a = pred.sum()
    b = target.sum()
    return (2.0 * intersection + eps) / (a + b + eps)

def iou_score(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return (intersection + eps) / (union + eps)

# ------------- MAIN -------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Dataset / Loader
    ds = BrainMetsDataset3DTest(TEST_ROOT, normalize=NORMALIZE)
    dl = DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=lambda batch: batch[0]) 


    # Modèle LearnOpenCV : 3 canaux in, 4 classes out
    model = UNet3D(in_channels=3, out_channels=4).to(DEVICE)

    # Charger le checkpoint et récupérer UNIQUEMENT les poids du modèle
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model"]         

    missing, unexpected = model.load_state_dict(state, strict=True)
    print("[INFO] Poids chargés depuis", CKPT_PATH)
    print(" - missing keys   :", missing)
    print(" - unexpected keys:", unexpected)


    model.eval()

    # Inférence
    for batch in dl:
        vol4   = batch["image"]          # (4, Z, Y, X)
        vol3   = vol4[:3, ...]           # (3, Z, Y, X)

        case_id = batch["case_id"]       # string
        ref_img = batch["ref_img"]       # Nifti1Image

        mask = sliding_window_infer_3d(
            model,
            vol3,
            roi_size=ROI_SIZE,
            overlap=OVERLAP,
            device=DEVICE,
            thr=THRESH
        )

        out_path = Path(OUT_DIR) / f"{case_id}_unet3d_mask.nii.gz"
        save_mask_nifti(mask, ref_img, out_path)
        print(f"[OK] {case_id}: {mask.shape} -> {out_path}")
    
    # ============================
    # ÉVALUATION DICE / IoU 
    # ============================
    print("\n[INFO] Début de l'évaluation Dice / IoU...")

    dices = []
    ious  = []

    # Fichier de résultats dans le dossier runs (un niveau au-dessus d'OUT_DIR)
    metrics_path = Path(OUT_DIR).parent / "metrics_dice_iou.csv"
    with open(metrics_path, "w") as f:
        f.write("case_id,dice,iou\n")

        # On parcourt tous les patients présents dans TEST_ROOT
        case_dirs = sorted([p for p in Path(TEST_ROOT).glob("*") if p.is_dir()])
        for base in case_dirs:
            case_id = base.name

            # Chemin du masque prédit
            pred_path = Path(OUT_DIR) / f"{case_id}_unet3d_mask.nii.gz"
            if not pred_path.exists():
                print(f"[WARN] Pas de prédiction trouvée pour {case_id}, on saute.")
                continue

            # Chemin de la GT neurochir 
            seg_candidates = list(base.glob("seg*.nii*"))
            if not seg_candidates:
                print(f"[WARN] Pas de seg.nii.gz pour {case_id}, on saute.")
                continue
            seg_path = seg_candidates[0]

            # Charger prédiction et GT (load_nii remet bien en (Z,Y,X))
            pred_zyx, _ = load_nii(pred_path)
            gt_zyx, _   = load_nii(seg_path)

            # Binariser la GT (tumeur vs fond)
            gt_bin = (gt_zyx > 0).astype(np.uint8)
            pred_bin = (pred_zyx > 0).astype(np.uint8)

            d = dice_coefficient(pred_bin, gt_bin)
            j = iou_score(pred_bin, gt_bin)

            dices.append(d)
            ious.append(j)

            f.write(f"{case_id},{d:.6f},{j:.6f}\n")
            print(f"[EVAL] {case_id}: Dice={d:.4f} | IoU={j:.4f}")

    if dices:
        dices = np.array(dices)
        ious  = np.array(ious)
        print("\n====== RÉSUMÉ GLOBAL ======")
        print(f"Dice moyen : {dices.mean():.4f} ± {dices.std():.4f}")
        print(f"IoU moyen  : {ious.mean():.4f} ± {ious.std():.4f}")
        print(f"[INFO] Résultats détaillés dans : {metrics_path}")
    else:
        print("[INFO] Aucune métrique calculée (aucun couple prédiction / seb.nii.gz trouvé).")



if __name__ == "__main__":
    main()

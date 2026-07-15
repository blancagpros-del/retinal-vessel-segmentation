# -*- coding: utf-8 -*-
"""
Pipeline aplicat a les 20 imatges d'entrenament de DRIVE (21 a 40).
Seguim el paper de Ramos-Soto et al. (2021) pas a pas.
Cada bloc de codi diu de quina seccio del paper ve, i quin lab de
l'assignatura fem servir per fer-ho (Fourier/Gaussia = Lab 5,
morfologia = Lab 8, binaritzacio = Lab 3).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as mr
from skimage.filters import gaussian as skgaussian
from scipy.ndimage import median_filter, convolve, rotate
from scipy.special import gamma
import tifffile

print("Llibreries carregades", flush=True)
plt.close('all')  # tanquem finestres d'execucions anteriors


def normalizar_robusto(x, p_bajo=1, p_alto=99):
    """
    Escala una imatge entre 0 i 1.
    Fem servir percentils en lloc del minim/maxim real,
    perque uns pocs pixels estranys no espatllin tota l'escala.
    """
    lo, hi = np.percentile(x[mask_fov], [p_bajo, p_alto])
    x_recortado = np.clip(x, lo, hi)
    return (x_recortado - lo) / (hi - lo)


def optimized_tophat(img, r_open, r_close):
    """
    Top-hat optimitzat (Seccio 3.2.1 del paper, Equacio 6).
    Fem servir obertura i tancament del Lab 8 (morfologia matematica).
    Primer invertim la imatge, despres obrim (treu els vasos petits),
    despres tanquem (neteja el fons), i restem per quedar-nos amb
    nomes els vasos.
    """
    img_real = np.real(img)
    img_norm = normalizar_robusto(img_real)
    Ic = 1.0 - img_norm  # imatge invertida (Equacio 7-8 del paper)
    So = mr.disk(r_open, decomposition='sequence')   # disc per obrir
    Sc = mr.disk(r_close, decomposition='sequence')  # disc per tancar
    opened = mr.opening(Ic, So)      # Lab 8: obertura = erosio + dilatacio
    closed = mr.closing(opened, Sc)  # Lab 8: tancament = dilatacio + erosio
    T = Ic - closed
    return normalizar_robusto(T)


def homomorphic_filter(img, sigma):
    """
    Filtre homomorfic (Seccio 3.2.2 del paper, Equacions 9-16).
    Fem servir Fourier del Lab 5 (transformades de Fourier i filtratge
    espacial): passem la imatge al domini de la frequencia, apliquem
    un filtre pas-alt gaussia, i tornem enrere.
    """
    z = np.log(img + 1e-6)  # domini logaritmic (Equacio 10)
    Fz = np.fft.fftshift(np.fft.fft2(z))  # Lab 5: transformada de Fourier
    N, M = np.shape(z)
    u, v = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, N))
    H = 1 - np.exp(-(u**2 + v**2) / (2 * sigma**2))  # filtre pas-alt (Equacio 12)
    im_filt = np.fft.ifft2(np.fft.ifftshift(Fz * H))  # tornem a l'espai normal
    g = np.real(im_filt)
    new_img = np.exp(g)  # desfem el logaritme (Equacio 16)
    return normalizar_robusto(new_img)


# MCET-HHO (Seccio 3.3.2 del paper).
# Buscar 4 llindars optims minimitzant l'entropia creuada,
# amb l'algorisme Harris Hawks Optimization (HHO).

def entropia_cruzada_multinivel(th, hist, niveles, termino_constante):
    """
    Diu com de bons son uns llindars th.
    Com mes petit el resultat, millor separen fons i vasos.
    Formula exacta: Equacions 20-21 del paper.
    """
    th_ordenados = sorted(int(round(np.clip(t, 1, len(niveles)))) for t in th)
    limites = [1] + th_ordenados + [len(niveles) + 1]
    suma_H = 0.0
    for k in range(len(limites) - 1):
        a, b = limites[k], limites[k + 1]
        if b <= a:
            continue
        cuentas = hist[a - 1:b - 1]  # pixels en aquest tram de grisos
        lv = niveles[a - 1:b - 1]
        total_cuentas = cuentas.sum()
        if total_cuentas <= 0:
            continue
        mu = (lv * cuentas).sum() / total_cuentas  # mitjana del tram
        mu = max(mu, 1.0)  # evitem log(0)
        suma_H += (lv * cuentas * np.log(mu)).sum()
    return termino_constante - suma_H


def vuelo_levy(dim, beta=1.5):
    """
    Salt aleatori tipus "vol de Levy".
    Es fa servir dins de HHO per explorar una mica mes lluny
    quan un halco fa un capbussament cap a la presa.
    """
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1 / beta))


def harris_hawks_optimization(funcion_fitness, dim, lb, ub, n_halcones=30, n_iter=250):
    """
    Metaheuristica HHO. Un grup d'halcons "cacen" la millor solucio.
    A cada iteracio es mouen cap al millor punt trobat fins ara,
    de manera mes agressiva o mes exploratoria segons una energia
    aleatoria E que va baixant amb el temps.
    Parametres iguals que el paper: 250 iteracions, 30 halcons.
    """
    X = np.random.uniform(lb, ub, size=(n_halcones, dim))  # posicio inicial, a l'atzar
    fitness = np.array([funcion_fitness(x) for x in X])
    idx_mejor = np.argmin(fitness)
    conejo = X[idx_mejor].copy()       # millor solucio trobada (el "conill")
    fitness_conejo = fitness[idx_mejor]

    for t in range(n_iter):
        E1 = 2 * (1 - t / n_iter)  # baixa amb el temps
        for i in range(n_halcones):
            E0 = 2 * np.random.rand() - 1
            E = E1 * E0
            q = np.random.rand()

            if abs(E) >= 1:
                # Energia alta: explorem molt, lluny del conill
                if q >= 0.5:
                    X_rand = X[np.random.randint(n_halcones)]
                    X[i] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * X[i])
                else:
                    X[i] = (conejo - X.mean(axis=0)) - np.random.rand() * (lb + np.random.rand() * (ub - lb))
            else:
                # Energia baixa: ja som a prop, ataquem
                r = np.random.rand()
                J = 2 * (1 - np.random.rand())
                if r >= 0.5 and abs(E) >= 0.5:
                    X[i] = (conejo - X[i]) - E * abs(J * conejo - X[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    X[i] = conejo - E * abs(conejo - X[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    Y = np.clip(conejo - E * abs(J * conejo - X[i]), lb, ub)
                    if funcion_fitness(Y) < fitness[i]:
                        X[i] = Y
                    else:
                        Z = np.clip(Y + np.random.rand(dim) * vuelo_levy(dim), lb, ub)
                        if funcion_fitness(Z) < fitness[i]:
                            X[i] = Z
                else:
                    Y = np.clip(conejo - E * abs(J * conejo - X.mean(axis=0)), lb, ub)
                    if funcion_fitness(Y) < fitness[i]:
                        X[i] = Y
                    else:
                        Z = np.clip(Y + np.random.rand(dim) * vuelo_levy(dim), lb, ub)
                        if funcion_fitness(Z) < fitness[i]:
                            X[i] = Z

            X[i] = np.clip(X[i], lb, ub)  # que no surti del rang valid

        # Actualitzem el millor si n'hem trobat un de millor
        fitness = np.array([funcion_fitness(x) for x in X])
        idx_mejor = np.argmin(fitness)
        if fitness[idx_mejor] < fitness_conejo:
            fitness_conejo = fitness[idx_mejor]
            conejo = X[idx_mejor].copy()

    return conejo, fitness_conejo


# --- Kernel del matched filter: nomes cal calcular-lo un cop ---
# Lab 5: convolucio d'una imatge amb un filtre (scipy.ndimage.convolve)
n_angulos = 26
sigma_mf = 0.8
size_mf = 7
eje = np.arange(size_mf) - size_mf // 2
x, y = np.meshgrid(eje, eje)
kernel_base = np.exp(-(x**2) / (2 * sigma_mf**2))
kernel_base = kernel_base - kernel_base.mean()

base = r"C:\Users\Usuario\Desktop\Física\PIVA\training"
numeros = list(range(21, 41))  # les 20 imatges d'entrenament
resultados = []

for n in numeros:
    print(f"\n=== Processant imatge {n} ===", flush=True)

    ruta_img = base + rf"\images\{n}_training.tif"
    ruta_mask = base + rf"\mask\{n}_training_mask.gif"
    ruta_gt = base + rf"\1st_manual\{n}_manual1.gif"

    imagen_drive = tifffile.imread(ruta_img)
    mask_fov = plt.imread(ruta_mask) > 0
    # Erosionem la FOV: la vora del disc sempre dona soroll fals.
    # Truc nostre, el paper no en parla.
    mask_fov_recorte = mr.binary_erosion(mask_fov, mr.disk(22))
    gt = plt.imread(ruta_gt) > 0

    # --- Preprocessat (Seccio 3.1): canal verd + Gaussia (Lab 5) ---
    im = imagen_drive[:, :, 1].astype(np.float64) / 255.0
    valor_medio_fov = im[mask_fov].mean()
    im_rellena = np.where(mask_fov, im, valor_medio_fov)  # omplim el fons, truc nostre
    G = skgaussian(im_rellena, sigma=np.sqrt(0.463))  # sigma exacte del paper

    # --- Rama gruixuda (Seccio 3.2) ---
    oth_thick = optimized_tophat(G, 8, 16)
    homo_thick = homomorphic_filter(oth_thick, 2)
    mediana_thick = median_filter(homo_thick, size=2)  # Lab 3: filtre de mediana
    refinado_thick = optimized_tophat(mediana_thick, 32, 86)

    # El paper no diu quin metode de binaritzacio fa servir aqui.
    # Fem servir un percentil (Lab 3: binaritzacio), triat a ma.
    umbral_thick = np.percentile(refinado_thick[mask_fov], 20)
    mask_thick = (refinado_thick < umbral_thick) & mask_fov_recorte

    # --- Rama fina (Seccio 3.3) ---
    oth_thin = optimized_tophat(G, 4, 20)
    homo_thin = homomorphic_filter(oth_thin, 20)

    # Matched filter (Seccio 3.3.1): girem el filtre en 26 angles
    # i ens quedem amb la resposta mes forta a cada pixel (Lab 5: convolucio)
    respuesta_max = np.full(homo_thin.shape, -np.inf)
    for theta in np.linspace(0, 182, n_angulos, endpoint=False):
        kernel_rot = rotate(kernel_base, theta, reshape=False, order=1)
        respuesta = convolve(homo_thin, kernel_rot, mode="reflect")
        respuesta_max = np.maximum(respuesta_max, respuesta)

    matched_thin = normalizar_robusto(respuesta_max)

    # --- MCET-HHO real (Seccio 3.3.2) ---
    img_niveles = (matched_thin * 255).astype(np.uint8)
    hist, _ = np.histogram(img_niveles[mask_fov], bins=256, range=(0, 256))
    niveles = np.arange(1, 257)
    termino_constante = np.sum(niveles * hist * np.log(niveles))

    def fitness_mcet(th):
        return entropia_cruzada_multinivel(th, hist, niveles, termino_constante)

    umbrales_opt, fitness_opt = harris_hawks_optimization(
        fitness_mcet, dim=4, lb=1, ub=256, n_halcones=30, n_iter=250
    )
    umbrales_thin = sorted(int(round(t)) for t in umbrales_opt)
    print(f"    Llindars MCET-HHO: {umbrales_thin} (fitness={fitness_opt:.2f})", flush=True)

    # El paper no diu exactament quin dels 4 llindars cal fer servir.
    # Fem servir el penultim (provat a ma, es el que millor funciona).
    mask_thin = (img_niveles > umbrales_thin[-2]) & mask_fov_recorte

    # --- Post-processat (Seccio 3.4): unio + neteja + suavitzat ---
    combinada = mask_thick | mask_thin  # unio OR de les dues rames
    sin_ruido = mr.remove_small_objects(combinada, min_size=30)  # Lab 8: treu soroll petit
    mask_final = mr.binary_closing(sin_ruido, mr.disk(2))  # Lab 8: tancament = uneix trossos
    mask_final = mask_final & mask_fov_recorte

    # --- Metriques (Seccio 4, Equacions 23-25), nomes dins la FOV ---
    TP = np.sum(mask_final & gt & mask_fov)
    TN = np.sum(~mask_final & ~gt & mask_fov)
    FP = np.sum(mask_final & ~gt & mask_fov)
    FN = np.sum(~mask_final & gt & mask_fov)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensibilidad = TP / (TP + FN)
    especificidad = TN / (TN + FP)
    dice = 2 * TP / (2 * TP + FP + FN)

    print(f"    Accuracy:      {accuracy:.4f}")
    print(f"    Sensibilitat:  {sensibilidad:.4f}")
    print(f"    Especificitat: {especificidad:.4f}")
    print(f"    Dice:          {dice:.4f}")

    resultados.append({
        "n": n, "accuracy": accuracy, "sensibilidad": sensibilidad,
        "especificidad": especificidad, "dice": dice,
        "mask_final": mask_final, "gt": gt
    })

    # Guardem la figura a disc, no la mostrem (amb 20 imatges saturaria Spyder)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1); plt.imshow(imagen_drive); plt.title(f"Original {n}")
    plt.subplot(1, 3, 2); plt.imshow(mask_final, cmap="gray"); plt.title(f"Mascara final (Dice={dice:.3f})")
    plt.subplot(1, 3, 3); plt.imshow(gt, cmap="gray"); plt.title("Ground truth")
    plt.tight_layout()
    plt.savefig(f"resultado_paper_drive_{n}.png", dpi=100)
    plt.close()

# RESUM FINAL: taula completa + mitjana i desviacio estandard
print("\n=== RESUM COMPLET (20 imatges, pipeline fidel al paper) ===", flush=True)
for r in resultados:
    print(f"Imatge {r['n']}: Dice={r['dice']:.4f}  Sens={r['sensibilidad']:.4f}  "
          f"Espec={r['especificidad']:.4f}  Acc={r['accuracy']:.4f}")

dice_vals = np.array([r["dice"] for r in resultados])
sens_vals = np.array([r["sensibilidad"] for r in resultados])
espec_vals = np.array([r["especificidad"] for r in resultados])
acc_vals = np.array([r["accuracy"] for r in resultados])

print("\n=== MITJANA +/- DESVIACIO ESTANDARD ===")
print(f"Accuracy:      {acc_vals.mean():.4f} +/- {acc_vals.std():.4f}")
print(f"Sensibilitat:  {sens_vals.mean():.4f} +/- {sens_vals.std():.4f}")
print(f"Especificitat: {espec_vals.mean():.4f} +/- {espec_vals.std():.4f}")
print(f"Dice:          {dice_vals.mean():.4f} +/- {dice_vals.std():.4f}")

print("\nACABAT SENSE ERRORS - LOT COMPLET FIDEL AL PAPER", flush=True)

# --- Resum visual: graella 4x5 amb les 20 mascares finals ---
plt.figure(figsize=(18, 15))
for i, r in enumerate(resultados):
    plt.subplot(4, 5, i + 1)
    plt.imshow(r["mask_final"], cmap="gray")
    plt.title(f"{r['n']} (Dice={r['dice']:.2f})", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("resumen_20_mascaras_paper.png", dpi=100)
plt.show()

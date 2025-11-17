import matplotlib.pyplot as plt
import os
import seaborn as sns

path = "plots"

#path = "ResNet-project/plots"


def plot_train_val_metric(results_cnn, results_res, metric_name, ylabel, outdir):
    """
    Version améliorée avec seaborn.
    - Style professionnel
    - Axe X avec entiers uniquement
    - Légende externe
    """

    sns.set_theme(style="whitegrid")  # thème propre et moderne

    epochs = range(1, len(results_cnn['train_' + metric_name]) + 1)

    plt.figure(figsize=(11, 6))

    # Couleurs cohérentes seaborn
    color_cnn = sns.color_palette("deep")[0]  # bleu
    color_res = sns.color_palette("deep")[3]  # rouge

    # --- CNN ---
    sns.lineplot(x=epochs, y=results_cnn['train_' + metric_name],
                 label=f'CNN — Train', color=color_cnn, marker='o')
    sns.lineplot(x=epochs, y=results_cnn['val_' + metric_name],
                 label=f'CNN — Validation', color=color_cnn, marker='o', linestyle='--')

    # --- ResNet ---
    sns.lineplot(x=epochs, y=results_res['train_' + metric_name],
                 label=f'ResNet — Train', color=color_res, marker='s')
    sns.lineplot(x=epochs, y=results_res['val_' + metric_name],
                 label=f'ResNet — Validation', color=color_res, marker='s', linestyle='--')

    # Labels − Titre
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"Performance Comparison — {metric_name.upper()}",
              fontsize=18, weight='bold')

    # Axe X : uniquement des entiers, pas de 0.5 !
    plt.xticks(ticks=list(epochs), labels=[str(e) for e in epochs])

    # Légende externe
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    plt.tight_layout()

    outfile = os.path.join(outdir, f"train_val_{metric_name}_seaborn.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()






def plot_test_bars(results_cnn, results_res, outdir):
    """
    Sauvegarde des graphes en barres pour test_loss et test_acc
    """

    models = ['CNN', 'ResNet']

    # --- Test Accuracy ---
    plt.figure(figsize=(7, 4))
    test_accs = [results_cnn['test_acc'], results_res['test_acc']]
    plt.bar(models, test_accs)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy — CNN vs ResNet")
    plt.tight_layout()

    outfile = os.path.join(outdir, "test_accuracy.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

    # --- Test Loss ---
    plt.figure(figsize=(7, 4))
    test_losses = [results_cnn['test_loss'], results_res['test_loss']]
    plt.bar(models, test_losses)
    plt.ylabel("Test Loss")
    plt.title("Test Loss — CNN vs ResNet")
    plt.tight_layout()

    outfile = os.path.join(outdir, "test_loss.png")
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_all(results_cnn, results_res, outdir="plots"):
    """
    Fonction globale pour sauver tous les graphiques en PNG.
    """

    # Create output directory if needed
    os.makedirs(outdir, exist_ok=True)

    print(f"Sauvegarde des graphiques dans : {outdir}")

    plot_train_val_metric(results_cnn, results_res, metric_name="loss", ylabel="Loss", outdir=outdir)
    plot_train_val_metric(results_cnn, results_res, metric_name="acc", ylabel="Accuracy", outdir=outdir)
    plot_test_bars(results_cnn, results_res, outdir=outdir)

    print("Tous les graphiques ont été enregistrés !")



import pickle

results_cnn = pickle.load(open("results/results_cnn.pkl", "rb"))
results_res = pickle.load(open("results/results_res.pkl", "rb"))

#results_cnn = pickle.load(open("ResNet-Project/results/results_cnn.pkl", "rb"))
#results_res = pickle.load(open("ResNet-Project/results/results_res.pkl", "rb"))

plot_all(results_cnn, results_res, outdir=path)
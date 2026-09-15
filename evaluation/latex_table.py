import csv
import os


def scappa_caratteri_latex(testo):
    """Esegue l'escape dei caratteri speciali di LaTeX."""
    caratteri_speciali = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        # "\\": "\\textbackslash{}",
    }
    for char, escape in caratteri_speciali.items():
        testo = testo.replace(char, escape)
    return testo


def converti_csv_in_latex(file_path):
    """Legge un CSV e restituisce la stringa del blocco tabular LaTeX."""
    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            dati = list(reader)
    except Exception as e:
        return f"\\textbf{{Errore nella lettura di {os.path.basename(file_path)}: {e}}}"

    if not dati:
        return "\\textit{File CSV vuoto.}"

    # Trasponi la matrice di dati se c'è al massimo una riga di dati effettivi (header + riga)
    if len(dati) <= 2:
        dati = list(map(list, zip(*dati)))

    num_colonne = len(dati[0])
    formato_colonne = "|" + "|".join(["c"] * num_colonne) + "|"

    linee_latex = []
    linee_latex.append(f"\\begin{{tabular}}{{{formato_colonne}}}")
    linee_latex.append("\\hline")

    for riga in dati:
        riga_pulita = [scappa_caratteri_latex(str(cella).strip()) for cella in riga]
        riga_formattata = " & ".join(riga_pulita) + " \\\\ \\hline"
        linee_latex.append(riga_formattata)

    linee_latex.append("\\end{tabular}")

    return "\n".join(linee_latex)


def genera_file_tex(lista_tabelle, file_output):
    """Genera il file .tex con caption e label lette dai dizionari."""
    with open(file_output, mode="w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\geometry{a4paper, margin=1in}\n")
        f.write("\\begin{document}\n\n")

        for tab in lista_tabelle:
            path = tab["path"]
            caption = tab.get("caption", f"Dati dal file {os.path.basename(path)}")
            label = tab.get("label", "")

            nome_file = os.path.basename(path)
            f.write(f"\\section*{{Tabella: {scappa_caratteri_latex(nome_file)}}}\n")

            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{scappa_caratteri_latex(caption)}}}\n")

            # Il label deve essere inserito rigorosamente dopo la caption
            if label:
                f.write(f"\\label{{{label}}}\n")

            f.write(converti_csv_in_latex(path) + "\n")
            f.write("\\end{table}\n\n")
            f.write("\\vspace{1cm}\n\n")

        f.write("\\end{document}\n")


# ==========================================
# Esempio di utilizzo
# ==========================================
if __name__ == "__main__":
    miei_csv = [
        "table_S0_summary.csv",
        "table_S1_example.csv",
        "table_S1_distribution.csv",
        "table_S2_step_evolution.csv",
        "table_S2_rar_summary.csv",
        "table_S2_adaptation_example.csv",
        "table_S3_no_route.csv",
        "table_cross_scenario_rsr.csv",
        "table_latency_summary.csv",
        "table_master_summary.csv",
    ]
    # La lista ora contiene dizionari con path, caption e label
    mie_tabelle = [
        {
            "path": "table_S0_summary.csv",
            "caption": "Summary S0)",
            "label": "tab:s0-summary",
        },
        {
            "path": "table_S1_example.csv",
            "caption": "Illustrative example",
            "label": "tab:s1-illustrative-example",
        },
        {
            "path": "table_S1_distribution.csv",
            "caption": "SPCO distribution across the origins affected by the hazard",
            "label": "tab:s1-spco-distribution",
        },
        {
            "path": "table_S2_step_evolution.csv",
            "caption": "Step-by-step evolution",
            "label": "tab:s2-step-evolution",
        },
        {
            "path": "table_S2_rar_summary.csv",
            "caption": "Route Adaptation Rate per Transition + Overall Row",
            "label": "tab:s2-route-adaptation-rate",
        },
        {
            "path": "table_S2_adaptation_example.csv",
            "caption": "Concrete examples of adaptation",
            "label": "tab:s2-adaptation-examples",
        },
        {
            "path": "table_S3_no_route.csv",
            "caption": "Origins left without a safe course",
            "label": "tab:s3-no-route",
        },
        {
            "path": "table_cross_scenario_rsr.csv",
            "caption": "Comparison of RSR Between S0, S1, and S3",
            "label": "tab:cross-scenario-rsr",
        },
        {
            "path": "table_latency_summary.csv",
            "caption": "Average/maximum routing latency by scenario",
            "label": "tab:latency-summary",
        },
        {
            "path": "table_master_summary.csv",
            "caption": "Cross-Scenario Summary Table",
            "label": "tab:master-summary",
        },
    ]

    file_destinazione = "tabelle_generate.tex"

    genera_file_tex(mie_tabelle, file_destinazione)
    print(f"File LaTeX generato con successo: {file_destinazione}")

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
OUTPUT = ROOT / "Final_Research_Report.pdf"


def load_json(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def add_footer(fig: plt.Figure, page: int) -> None:
    fig.text(
        0.5,
        0.018,
        f"Graph-Structure-Aware Hyperbox Refinement | Final research report | {page}",
        ha="center",
        fontsize=8,
        color="#667085",
    )


def text_page(pdf: PdfPages, title: str, paragraphs: list[str], page: int, subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.09, 0.93, title, fontsize=22, weight="bold", color="#16324F")
    if subtitle:
        fig.text(0.09, 0.895, subtitle, fontsize=10, color="#667085")
    y = 0.85 if subtitle else 0.89
    for paragraph in paragraphs:
        if paragraph.startswith("## "):
            fig.text(0.09, y, paragraph[3:], fontsize=14, weight="bold", color="#1F5A7A")
            y -= 0.045
            continue
        lines = textwrap.wrap(paragraph, width=92)
        fig.text(0.09, y, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.45, color="#202124")
        y -= 0.035 * len(lines) + 0.035
    add_footer(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(
    pdf: PdfPages,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    page: int,
    note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.09, 0.93, title, fontsize=22, weight="bold", color="#16324F")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.06, 0.55, 0.88, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row == 0:
            cell.set_facecolor("#16324F")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "white")
    if note:
        fig.text(
            0.09,
            0.48,
            "\n".join(textwrap.wrap(note, width=92)),
            fontsize=10,
            va="top",
            linespacing=1.45,
            color="#344054",
        )
    add_footer(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def figure_page(pdf: PdfPages, title: str, image_names: list[str], page: int, captions: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.09, 0.94, title, fontsize=22, weight="bold", color="#16324F")
    count = len(image_names)
    for index, name in enumerate(image_names):
        image = mpimg.imread(FIGURES / name)
        if count == 1:
            rect = [0.08, 0.15, 0.84, 0.70]
        else:
            rect = [0.08, 0.53 - index * 0.42, 0.84, 0.30]
        ax = fig.add_axes(rect)
        ax.imshow(image)
        ax.axis("off")
        fig.text(rect[0], rect[1] - 0.025, captions[index], fontsize=9, color="#475467")
    add_footer(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def mean_rows(data: dict, variants: list[str]) -> list[list[str]]:
    rows = []
    for name in variants:
        item = data["aggregate"][name]
        rows.append(
            [
                name.capitalize(),
                f'{item["mean_test_accuracy"]:.4f} ± {item["test_accuracy_std"]:.4f}',
                f'{item["mean_test_macro_f1"]:.4f} ± {item["test_macro_f1_std"]:.4f}',
                f'{item["box_count_mean"]:.1f}',
                f'{item["test_accuracy_delta_vs_baseline"]:+.4f}',
            ]
        )
    return rows


def ablation_rows(data: dict) -> list[list[str]]:
    labels = {
        "E1": "GCN softmax",
        "E2": "GCN + FMNN",
        "E3": "Geometry merge",
        "E4": "Graph-aware merge",
        "E5": "Pruning only",
        "E6": "Full GSHR",
        "E7": "Graph-disabled full",
        "E8": "Unsafe full",
    }
    rows = []
    for item in data["experiments"]:
        results = [run["selected_result"] for run in item["runs"]]
        accuracy = sum(result["accuracy"] for result in results) / len(results)
        macro_f1 = sum(result["macro_f1"] for result in results) / len(results)
        boxes = sum(result.get("box_count", result.get("num_hyperboxes", 0)) for result in results) / len(results)
        rows.append([item["name"], labels[item["name"]], f"{accuracy:.4f}", f"{macro_f1:.4f}", f"{boxes:.1f}"])
    return rows


def main() -> None:
    cora = load_json("final_evaluation_current.json")
    citeseer = load_json("final_evaluation_citeseer_current.json")
    ablation = load_json("ablation_summary_corrected.json")
    with PdfPages(OUTPUT) as pdf:
        page = 1
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("#F7FAFC")
        fig.text(0.09, 0.70, "Graph-Structure-Aware\nHyperbox Refinement", fontsize=30, weight="bold", color="#16324F")
        fig.text(0.09, 0.57, "for Graph Neural Network-Based Node Classification", fontsize=16, color="#1F5A7A")
        fig.text(0.09, 0.45, "Final Research Report", fontsize=18, weight="bold", color="#344054")
        fig.text(0.09, 0.38, "Cora and CiteSeer evaluation\nFive-seed validation-safe experimental study", fontsize=12, color="#667085", linespacing=1.5)
        fig.text(0.09, 0.12, "Generated: 6 November 2026\nProject: gnn-fmnn-gshr", fontsize=10, color="#667085")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        text_page(
            pdf,
            "Abstract and Executive Summary",
            [
                "This study evaluates a graph-structure-aware hyperbox refinement (GSHR) pipeline for node classification. A two-layer graph convolutional network (GCN) produces node embeddings, a fuzzy min-max neural network (FMNN) converts training embeddings into class-labelled hyperboxes, and graph-aware merging and pruning attempt to compact the representation without using held-out test labels.",
                "The primary experiments use the public Planetoid splits for Cora and CiteSeer. Hyperparameters are selected using validation data only, and final results are averaged across five random seeds. The current evidence shows that refinement reduces the mean number of hyperboxes, but predictive gains are not consistent. On current Cora results, full refinement reduces the mean box count from 12.2 to 9.0 while reducing accuracy by 0.96 percentage points. On CiteSeer, the mean box count falls from 9.8 to 7.8 while accuracy remains almost unchanged.",
                "The supported conclusion is therefore that GSHR is a compactness and interpretability mechanism rather than a demonstrated universal accuracy improvement. Further evaluation on more datasets, splits, and computational-cost measures is required.",
            ],
            page,
            "Final report based on the current authoritative result artifacts",
        )
        page += 1
        text_page(
            pdf,
            "1. Introduction and Research Questions",
            [
                "Graph neural networks use graph connectivity and node attributes to learn representations for node classification. Although effective, their learned representations do not directly provide compact, human-readable decision regions. Fuzzy min-max neural networks offer an alternative representation based on axis-aligned hyperboxes, making them suitable for prototype-style inspection and compression.",
                "This project combines a GCN encoder with an FMNN classifier and introduces graph-structure-aware refinement. The refinement layer uses hyperbox geometry, graph cohesion, neighborhood label agreement, validation purity, and a validation objective to decide whether boxes should be merged or pruned.",
                "## Research questions",
                "RQ1. Can FMNN hyperboxes classify GCN-generated node embeddings on standard citation graphs?\nRQ2. Can graph-aware refinement reduce the number of hyperboxes?\nRQ3. Does refinement preserve or improve accuracy and macro-F1?\nRQ4. Does a configuration selected on Cora transfer to CiteSeer without retuning?",
                "## Contributions",
                "The project contributes a modular GCN-FMNN pipeline, graph-aware hyperbox refinement, validation-safe candidate acceptance, explicit split and leakage checks, corrected ablation reporting, and a frozen cross-dataset evaluation protocol.",
            ],
            page,
        )
        page += 1
        text_page(
            pdf,
            "2. Methodology",
            [
                "## GCN encoder",
                "The encoder is a two-layer GCN with a 64-dimensional hidden layer, ReLU activation, dropout of 0.5, Adam optimization, learning rate 0.01, weight decay 0.0005, a maximum of 200 epochs, and validation-based early stopping with patience 30. The hidden layer is used as the node embedding representation.",
                "## FMNN classification",
                "Embeddings are min-max normalized using statistics fitted only on training nodes. The FMNN is fitted only on training embeddings and labels. Each hyperbox stores lower and upper bounds, a class label, and its member nodes. The selected parameters are theta = 0.5 and gamma = 1.0.",
                "## GSHR refinement",
                "Merging is restricted to same-class boxes and is constrained by maximum side length, volume inflation, graph cohesion, validation purity, and the validation objective. Pruning requires class coverage, purity, cohesion, and a strict improvement in the validation objective. The objective combines validation accuracy and macro-F1 with equal weight.",
                "Test data are not used to select hyperparameters or accept refinement candidates. CiteSeer is evaluated using the configuration selected on Cora and is not retuned.",
            ],
            page,
        )
        page += 1
        table_page(
            pdf,
            "3. Datasets and Experimental Configuration",
            ["Dataset", "Nodes", "Features", "Classes", "Split"],
            [
                ["Cora", "2,708", "1,433", "7", "Public Planetoid"],
                ["CiteSeer", "3,327", "3,703", "6", "Public Planetoid"],
            ],
            page,
            "The implementation validates that train, validation, and test masks are disjoint. The same graph is available to the transductive GCN during message passing, while training labels are the only labels used to fit FMNN hyperboxes and graph-aware refinement.",
        )
        page += 1
        table_page(
            pdf,
            "4. Current Cora Results",
            ["Variant", "Accuracy (mean ± SD)", "Macro-F1 (mean ± SD)", "Mean boxes", "Δ accuracy"],
            mean_rows(cora, ["baseline", "merge", "prune", "full"]),
            page,
            "Five seeds. Values are held-out test metrics after validation-only configuration selection and refinement. Full refinement reduces the mean representation size by approximately 26.2% (12.2 to 9.0 boxes), but accuracy and macro-F1 are lower than the baseline.",
        )
        page += 1
        table_page(
            pdf,
            "5. Current CiteSeer Results",
            ["Variant", "Accuracy (mean ± SD)", "Macro-F1 (mean ± SD)", "Mean boxes", "Δ accuracy"],
            mean_rows(citeseer, ["baseline", "merge", "prune", "full"]),
            page,
            "The Cora-selected configuration is frozen for this evaluation. Full refinement reduces the mean box count by approximately 20.4% (9.8 to 7.8 boxes) while changing accuracy by only -0.06 percentage points.",
        )
        page += 1
        table_page(
            pdf,
            "6. E1–E8 Ablation Study",
            ["ID", "Variant", "Accuracy", "Macro-F1", "Mean boxes"],
            ablation_rows(ablation),
            page,
            "The corrected ablation artifact is authoritative. The softmax GCN is the strongest predictive baseline in this matrix. Graph-aware merging slightly improves over geometry-only merging, while full refinement provides compactness without a predictive improvement.",
        )
        page += 1
        figure_page(
            pdf,
            "7. Representation and Hyperbox Visualizations",
            ["cora_basic_fmnn_representation.png", "cora_gcn_fmnn_gshr_representation.png"],
            page,
            [
                "Figure 1. Cora FMNN representation.",
                "Figure 2. Cora GCN-FMNN-GSHR representation.",
            ],
        )
        page += 1
        figure_page(
            pdf,
            "8. Hyperbox Compactness and Ablation Figures",
            ["cora_hyperbox_representation.png", "compactness_vs_macro_f1.png"],
            page,
            [
                "Figure 3. Cora hyperbox representation.",
                "Figure 4. Compactness versus macro-F1.",
            ],
        )
        page += 1
        figure_page(
            pdf,
            "9. Ablation Figures",
            ["ablation_box_count.png", "ablation_macro_f1.png"],
            page,
            [
                "Figure 5. Mean hyperbox count across ablations.",
                "Figure 6. Macro-F1 across ablations.",
            ],
        )
        page += 1
        text_page(
            pdf,
            "10. Discussion",
            [
                "The experiments answer the compactness question positively: both datasets show fewer hyperboxes after merging/full refinement. However, the predictive objective is not consistently improved. On Cora, the current full-refinement result trades away accuracy and macro-F1 for a smaller representation. On CiteSeer, the same frozen configuration preserves accuracy more closely and slightly improves macro-F1 for merge/full refinement.",
                "The ablations indicate that geometry and graph structure both contribute to refinement decisions, but the graph-aware advantage over geometry-only merging is modest. The unsafe variant reinforces the importance of class-purity and validation safeguards. The softmax GCN remains the strongest predictive reference in the corrected ablation matrix.",
                "These results suggest that GSHR should be evaluated according to an explicit accuracy–compactness trade-off. It may be useful when representation size, prototype inspection, or decision-region visualization matters, even when it is not the highest-accuracy classifier.",
            ],
            page,
        )
        page += 1
        text_page(
            pdf,
            "11. Validity, Reproducibility, and Limitations",
            [
                "## Validity safeguards",
                "Split masks are checked for disjointness. Normalization is fitted on training embeddings. FMNN fitting uses training labels only. Refinement decisions use validation embeddings and labels. Test metrics are computed after configuration selection and refinement. CiteSeer is evaluated without retuning.",
                "## Limitations",
                "The study covers two citation datasets and five seeds. It uses public transductive splits and does not yet provide runtime, memory, significance, or inductive generalization analyses. Hyperbox behavior depends on the learned embedding geometry and selected FMNN parameters. Confidence intervals across seeds do not replace independent graph splits or broader datasets.",
                "## Reproduction",
                "The main commands are: python -m pytest -q; python scripts/tune_validation.py --config configs/experiments.yaml; python scripts/final_evaluation.py; python scripts/final_evaluation.py --experiments configs/citeseer_experiments.yaml; and python scripts/evaluate.py --results-dir results/. Source code, configurations, tests, tables, and figures are included in the repository.",
            ],
            page,
        )
        page += 1
        text_page(
            pdf,
            "12. Conclusion and Future Work",
            [
                "The project successfully integrates GCN embeddings, FMNN hyperboxes, and graph-structure-aware refinement in a reproducible node-classification pipeline. The current experiments demonstrate that refinement can reduce the number of hyperboxes on both Cora and CiteSeer.",
                "The evidence does not support a universal claim of improved predictive performance. The strongest conclusion is that GSHR offers a compactness and interpretability mechanism with a measurable accuracy–representation trade-off.",
                "Future work should evaluate PubMed and additional graph datasets, repeated splits, larger seed sets, runtime and memory cost, statistical significance, inductive settings, learned refinement thresholds, calibration, and comparisons with GraphSAGE, GAT, MLP, and other prototype-based classifiers.",
                "## References",
                "Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.\nGabriel et al., Learning from Non-Overlapping Hyperboxes in Fuzzy Min-Max Neural Networks, foundational FMNN literature.\nSen et al., Collective Classification in Network Data, the Cora/CiteSeer citation-network benchmark source.\nPyTorch Geometric Planetoid dataset documentation and implementation.",
            ],
            page,
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()



from dataclasses import dataclass
import os
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from source.pv.creation import PVSizingFilePathBuilder


@dataclass
class ViolinPlotterConfig:
    """Configuration for the violin plotter."""
    indicator_configs: list[tuple[str, str]]
    house_id: int
    save_to_pdf: bool


class ViolinPlotter:
    def __init__(self, config: ViolinPlotterConfig):
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self):
        df = self._load_data()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.set_style("whitegrid")

        # Plot 1: NPV
        sns.violinplot(
            x="metoda", y="npv", data=df,
            inner="quartile", palette="Set2", ax=axes[0, 0]
        )
        axes[0, 0].set_title("NPV distribution by method")
        axes[0, 0].set_xlabel("Method")
        axes[0, 0].set_ylabel("NPV")

        # Plot 2: Self-consumption
        sns.violinplot(
            x="metoda", y="self_consumption", data=df,
            inner="quartile", palette="Set2", ax=axes[0, 1]
        )
        axes[0, 1].set_title("Self-consumption distribution by method")
        axes[0, 1].set_xlabel("Method")
        axes[0, 1].set_ylabel("Self-consumption")

        # Plot 3: CAPEX
        sns.violinplot(
            x="metoda", y="capex", data=df,
            inner="quartile", palette="Set2", ax=axes[1, 0]
        )
        axes[1, 0].set_title("CAPEX distribution by method")
        axes[1, 0].set_xlabel("Method")
        axes[1, 0].set_ylabel("CAPEX")

        # Plot 4: NPV/CapEX ratio
        sns.violinplot(
            x="metoda", y="npv_capex_ratio", data=df,
            inner="quartile", palette="Set2", ax=axes[1, 1]
        )
        axes[1, 1].set_title("NPV/CapEX ratio distribution by method")
        axes[1, 1].set_xlabel("Method")
        axes[1, 1].set_ylabel("NPV/CapEX ratio")

        self._finalize_plot(fig, axes)

    def _load_data(self) -> pd.DataFrame:
        dfs = []
        for indicator_name, method_label in self._config.indicator_configs:
            csv_path = self._builder.get_economical_space_csv_path(
                indicator_name)
            df_indicator = pd.read_csv(csv_path)

            # Tag rows with the sizing method (indicator used)
            df_indicator["metoda"] = method_label

            # Ensure we have the NPV/CAPEX ratio even if not precomputed
            if ("npv" in df_indicator.columns and
                    "capex" in df_indicator.columns):
                capex_nonzero = df_indicator["capex"].replace(0, pd.NA)
                df_indicator["npv_capex_ratio"] = (
                    df_indicator["npv"] / capex_nonzero
                )

            dfs.append(df_indicator)

        df = pd.concat(dfs, ignore_index=True)
        return df

    def _finalize_plot(self, fig: Figure, axes: np.ndarray):
        plt.tight_layout()
        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            indicator_names = [indicator_name for indicator_name,
                               _ in self._config.indicator_configs]
            house_id = self._config.house_id
            file_name = f"violin_plot_{house_id}_{'_'.join(indicator_names)}.pdf"
            file_path = os.path.join(folder, file_name)
            plt.savefig(file_path)
        else:
            plt.show()


class Dominance3DPlotter:
    """
    3D plotter that visualises where the comparison method dominates
    the base method in NPV and/or self-consumption across the
    (pik, ick, ck) price space.
    """

    def __init__(self, config: DominancePlotterConfig):
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self) -> None:
        dom = DominancePlotter(self._config)
        df = dom._load_and_clean_data()
        if df.empty:
            print("No data available for 3D dominance plot.")
            return

        df_pair = (
            df.pivot_table(
                index=["pik", "ick", "ck"],
                columns="metoda",
                values=["npv", "self_consumption"],
                aggfunc="first",
            )
            .reset_index()
        )

        df_pair.columns = [
            f"{a}_{b}" if b else a
            for a, b in df_pair.columns
        ]

        base = self._config.base_label
        comp = self._config.comparison_label
        tol = self._config.tol

        npv_base_col = f"npv_{base}"
        npv_comp_col = f"npv_{comp}"
        sc_base_col = f"self_consumption_{base}"
        sc_comp_col = f"self_consumption_{comp}"

        required_cols = [
            npv_base_col,
            npv_comp_col,
            sc_base_col,
            sc_comp_col,
        ]
        if any(col not in df_pair.columns for col in required_cols):
            print(
                "Missing required columns in 3D dominance DataFrame: "
                f"{[c for c in required_cols if c not in df_pair.columns]}"
            )
            return

        both_higher = df_pair[
            (df_pair[npv_comp_col] - df_pair[npv_base_col] > tol)
            & (df_pair[sc_comp_col] - df_pair[sc_base_col] > tol)
        ]

        npv_only = df_pair[
            (df_pair[npv_comp_col] - df_pair[npv_base_col] > tol)
            & ~(
                (df_pair[sc_comp_col] - df_pair[sc_base_col] > tol)
            )
        ]

        sc_only = df_pair[
            (df_pair[sc_comp_col] - df_pair[sc_base_col] > tol)
            & ~(
                (df_pair[npv_comp_col] - df_pair[npv_base_col] > tol)
            )
        ]

        print(f"NPV higher only: {len(npv_only)}")
        print(f"SC higher only: {len(sc_only)}")
        print(f"Both NPV & SC higher: {len(both_higher)}")

        fig = plt.figure(figsize=(10, 7))
        ax: Any = fig.add_subplot(111, projection="3d")

        xs_all = np.asarray(df_pair["pik"])
        ys_all = np.asarray(df_pair["ick"])
        zs_all = np.asarray(df_pair["ck"])

        ax.scatter(  # type: ignore[arg-type]
            xs_all,
            ys_all,
            zs=zs_all,
            c="lightgray",
            alpha=0.3,
            label="All scenarios",
        )

        if not npv_only.empty:
            xs_npv = np.asarray(npv_only["pik"])
            ys_npv = np.asarray(npv_only["ick"])
            zs_npv = np.asarray(npv_only["ck"])

            ax.scatter(  # type: ignore[arg-type]
                xs_npv,
                ys_npv,
                zs=zs_npv,
                c="purple",
                s=50,
                label=f"{comp} NPV > {base} NPV",
            )

        if not sc_only.empty:
            xs_sc = np.asarray(sc_only["pik"])
            ys_sc = np.asarray(sc_only["ick"])
            zs_sc = np.asarray(sc_only["ck"])

            ax.scatter(  # type: ignore[arg-type]
                xs_sc,
                ys_sc,
                zs=zs_sc,
                c="yellow",
                s=50,
                label=f"{comp} SC > {base} SC",
            )

        if not both_higher.empty:
            xs_both = np.asarray(both_higher["pik"])
            ys_both = np.asarray(both_higher["ick"])
            zs_both = np.asarray(both_higher["ck"])

            ax.scatter(  # type: ignore[arg-type]
                xs_both,
                ys_both,
                zs=zs_both,
                c="blue",
                s=50,
                label=f"{comp} better in NPV & SC",
            )

        ax.set_xlabel("FiT")     # pik
        ax.set_ylabel("InstC")   # ick
        ax.set_zlabel("EnergC")  # ck
        ax.set_title(f"{comp} vs {base} dominance in price space")
        ax.legend()
        ax.view_init(elev=12, azim=260)

        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            house_id = self._config.house_id
            file_name = (
                "case_distribution_3d_"
                f"{house_id}_{self._config.base_indicator}_"
                f"{self._config.comparison_indicator}.pdf"
            )
            file_path = os.path.join(folder, file_name)
            plt.savefig(file_path)
        else:
            plt.show()

        plt.close(fig)

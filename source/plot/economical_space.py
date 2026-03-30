

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

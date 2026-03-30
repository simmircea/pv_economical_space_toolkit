from dataclasses import dataclass
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from source.plot.cost_space import (
    _house_index_map,
    _load_consumption_per_house,
)
from source.pv.creation import PVSizingFilePathBuilder


@dataclass
class DominancePlotterConfig:
    """
    Configuration for the dominance comparison plotter.

    Attributes:
        house_ids: Houses to plot; one figure per house. If None, plot
            all houses present in the economical-space CSVs.
        base_indicator: Name of the base indicator CSV (e.g. "neeg").
        base_label: Label used in plots for the base indicator.
        comparison_indicator: Name of the comparison indicator
            (e.g. "roi").
        comparison_label: Label used in plots for the comparison
            indicator.
        sample_size: Maximum number of price triples to show per figure.
        tol: Numerical tolerance when comparing NPV and self-consumption.
        save_to_pdf: If True, save figures; otherwise call plt.show().
    """

    house_ids: list[int] | None = None
    base_indicator: str = "neeg"
    base_label: str = "NEEG"
    comparison_indicator: str = "roi"
    comparison_label: str = "ROI"
    sample_size: int = 5
    tol: float = 1e-6
    save_to_pdf: bool = True


class DominancePlotter:
    """
    Plotter that compares two sizing methods (e.g. NEEG vs NPV/CapEX)
    for identical price triples and visualises when the comparison
    method dominates in NPV and/or self-consumption.
    """

    def __init__(self, config: DominancePlotterConfig):
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self):
        df_all_raw = self._load_and_clean_data()
        df_all = pd.DataFrame(df_all_raw)
        if df_all.empty:
            print("No data available for dominance comparison.")
            return

        consumption = _load_consumption_per_house(self._builder)
        index_map: pd.Series
        if consumption is not None:
            index_map = _house_index_map(consumption)
        else:
            if "house_id" in df_all.columns:
                unique_ids = sorted(
                    df_all["house_id"].dropna().astype(int).unique().tolist()
                )
                unique_houses = pd.DataFrame({"house_id": unique_ids})
                index_map = _house_index_map(
                    unique_houses,
                    consumption_col=None,
                )
            else:
                index_map = pd.Series(dtype=int)

        house_ids = self._config.house_ids
        if "house_id" not in df_all.columns:
            house_ids = None

        if house_ids is None:
            if "house_id" not in df_all.columns:
                # Fallback: aggregate over all houses (single figure).
                df_pair, df_both_cases, df_sc_only_cases = \
                    self._prepare_cases(df_all)
                if df_both_cases.empty and df_sc_only_cases.empty:
                    print(
                        "No dominance cases found for the given "
                        "configuration."
                    )
                    return

                both_title = (
                    f"{self._config.comparison_label} improves BOTH NPV "
                    f"and Self-consumption"
                )
                sc_only_title = (
                    f"{self._config.comparison_label} improves ONLY "
                    f"Self-consumption"
                )
                self._plot_2panel(
                    df_both_cases,
                    cm.get_cmap("Blues"),
                    both_title,
                    house_index=None,
                    house_id=None,
                )
                self._plot_2panel(
                    df_sc_only_cases,
                    cm.get_cmap("Reds"),
                    sc_only_title,
                    house_index=None,
                    house_id=None,
                )
                return

            house_ids = sorted(
                df_all["house_id"].dropna().astype(int).unique().tolist()
            )
        else:
            available = set(
                df_all["house_id"].dropna().astype(int).unique().tolist()
            )
            house_ids = sorted([h for h in house_ids if h in available])

        if not house_ids:
            print("No matching houses for dominance comparison.")
            return

        for house_id in house_ids:
            df_house = pd.DataFrame(
                df_all[df_all["house_id"] == house_id]
            ).copy()
            if df_house.empty:
                continue

            df_pair, df_both_cases, df_sc_only_cases = \
                self._prepare_cases(df_house)
            if df_pair.empty:
                continue

            if df_both_cases.empty and df_sc_only_cases.empty:
                continue

            house_index = (
                int(index_map.loc[house_id])
                if house_id in index_map.index
                else None
            )
            title_idx = house_index or house_id

            both_title = (
                f"{self._config.comparison_label} improves BOTH NPV and "
                f"Self-consumption (House {title_idx})"
            )
            sc_only_title = (
                f"{self._config.comparison_label} improves ONLY "
                f"Self-consumption (House {title_idx})"
            )

            self._plot_2panel(
                df_both_cases,
                cm.get_cmap("Blues"),
                both_title,
                house_index=house_index,
                house_id=house_id,
            )

            self._plot_2panel(
                df_sc_only_cases,
                cm.get_cmap("Reds"),
                sc_only_title,
                house_index=house_index,
                house_id=house_id,
            )

    def _load_and_clean_data(self):
        indicator_configs = [
            (self._config.base_indicator, self._config.base_label),
            (self._config.comparison_indicator,
             self._config.comparison_label),
        ]

        dfs: list[pd.DataFrame] = []
        for indicator_name, method_label in indicator_configs:
            csv_path = self._builder.get_economical_space_csv_path(
                indicator_name)
            if not os.path.exists(csv_path):
                continue

            df_indicator = pd.read_csv(csv_path)
            df_indicator["metoda"] = method_label

            if {"initial_cost_per_kW", "cost_per_kW",
                    "feed_in_price_per_kW"}.issubset(
                        df_indicator.columns):
                df_indicator["ick"] = df_indicator["initial_cost_per_kW"]
                df_indicator["ck"] = df_indicator["cost_per_kW"]
                df_indicator["pik"] = df_indicator["feed_in_price_per_kW"]

            dfs.append(df_indicator)

        if not dfs:
            return pd.DataFrame(
                columns=[
                    "metoda", "npv", "self_consumption", "capex",
                    "pik", "ick", "ck",
                ]
            )

        df = pd.concat(dfs, ignore_index=True)

        for col in ["npv", "self_consumption", "capex"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["pik", "ick", "ck"]:
            if col in df.columns:
                df[col] = df[col].round(4)

        df = df[df["metoda"].isin(
            [self._config.base_label,
             self._config.comparison_label]
        )].copy()

        return df

    def _prepare_cases(
        self,
        df: pd.DataFrame,
    ):
        if df.empty:
            return df, df, df

        df_pair = (
            df.pivot_table(
                index=["pik", "ick", "ck"],
                columns="metoda",
                values=["npv", "self_consumption", "capex"],
                aggfunc="first",
            )
            .reset_index()
        )

        df_pair.columns = [
            f"{a}_{b}" if b else a
            for a, b in df_pair.columns
        ]

        tol = self._config.tol
        base = self._config.base_label
        comp = self._config.comparison_label

        npv_base_col = f"npv_{base}"
        npv_comp_col = f"npv_{comp}"
        sc_base_col = f"self_consumption_{base}"
        sc_comp_col = f"self_consumption_{comp}"
        capex_base_col = f"capex_{base}"
        capex_comp_col = f"capex_{comp}"

        required_cols = [
            npv_base_col,
            npv_comp_col,
            sc_base_col,
            sc_comp_col,
            capex_base_col,
            capex_comp_col,
        ]

        if any(col not in df_pair.columns for col in required_cols):
            return df_pair, df_pair.iloc[0:0], df_pair.iloc[0:0]

        df_pair["better_npv"] = (
            df_pair[npv_comp_col] >
            df_pair[npv_base_col] + tol
        )

        df_pair["better_sc"] = (
            df_pair[sc_comp_col] >
            df_pair[sc_base_col] + tol
        )

        df_both_cases = self._safe_sample(
            df_pair[df_pair["better_npv"] & df_pair["better_sc"]]
        )
        df_sc_only_cases = self._safe_sample(
            df_pair[(~df_pair["better_npv"]) & df_pair["better_sc"]]
        )

        return df_pair, df_both_cases, df_sc_only_cases

    def _safe_sample(self, df_in):
        if df_in.empty:
            return df_in
        n = min(self._config.sample_size, len(df_in))
        return df_in.sample(n=n, random_state=39)

    def _plot_2panel(
        self,
        df_cases,
        cmap,
        title: str,
        house_id: int | None,
        house_index: int | None,
    ) -> None:
        if df_cases.empty:
            print(f"No cases available for: {title}")
            return

        base = self._config.base_label
        comp = self._config.comparison_label

        capex_base_col = f"capex_{base}"
        capex_comp_col = f"capex_{comp}"
        npv_base_col = f"npv_{base}"
        npv_comp_col = f"npv_{comp}"
        sc_base_col = f"self_consumption_{base}"
        sc_comp_col = f"self_consumption_{comp}"

        required_cols = [
            capex_base_col,
            capex_comp_col,
            npv_base_col,
            npv_comp_col,
            sc_base_col,
            sc_comp_col,
        ]
        if any(col not in df_cases.columns for col in required_cols):
            print(
                "Missing required columns in dominance DataFrame: "
                f"{[c for c in required_cols if c not in df_cases.columns]}"
            )
            return

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 10), sharex=True
        )

        colors = cmap(np.linspace(0.25, 0.95, len(df_cases)))
        price_legend_elements: list[Line2D] = []

        for (_, row), color in zip(df_cases.iterrows(), colors):
            x_capex = row[capex_comp_col]
            x_neeg = row[capex_base_col]

            ax1.scatter(
                x_capex,
                row[npv_comp_col],
                marker="*",
                s=220,
                color=color,
                edgecolor="k",
            )

            ax1.scatter(
                x_neeg,
                row[npv_base_col],
                marker="o",
                s=110,
                color=color,
                edgecolor="k",
            )

            ax2.scatter(
                x_capex,
                row[sc_comp_col],
                marker="*",
                s=220,
                color=color,
                edgecolor="k",
            )

            ax2.scatter(
                x_neeg,
                row[sc_base_col],
                marker="o",
                s=110,
                color=color,
                edgecolor="k",
            )

            label = (
                f"({row['pik']:.3f}, "
                f"{row['ick']:.3f}, "
                f"{row['ck']:.3f})"
            )

            price_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    linestyle="None",
                    markersize=8,
                    label=label,
                )
            )

        ax1.set_title("CAPEX vs NPV")
        ax1.set_ylabel("NPV")
        ax1.grid(True)

        ax2.set_title("CAPEX vs Self-consumption")
        ax2.set_xlabel("CAPEX")
        ax2.set_ylabel("Self-consumption")
        ax2.grid(True)

        method_legend = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                label=self._config.comparison_label,
                markerfacecolor="gray",
                markersize=14,
                markeredgecolor="k",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=self._config.base_label,
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="k",
            ),
        ]

        fig.legend(
            handles=method_legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=2,
        )

        fig.legend(
            handles=price_legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(price_legend_elements),
            title="Price scenarios (FiT, InstC, EnergC)",
        )

        fig.suptitle(title, fontsize=14)

        plt.subplots_adjust(bottom=0.22, top=0.90)

        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            filename = title.replace(" ", "_").replace("/", "_")
            suffix = f"house_{house_id}" if house_index is not None else ""
            file_path = os.path.join(folder, f"{filename}{suffix}.pdf")
            plt.savefig(file_path, bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)

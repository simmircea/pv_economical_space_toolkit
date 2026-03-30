"""
Cost-space plotters for economical-space evaluation results.

This module provides three plotters that use the economical-space CSVs
(economical_space_neeg.csv, economical_space_npv.csv, economical_space_roi.csv)
and optionally pv_statistics.csv (for total_consumption_kwh per house):

1) ConsumptionStatsPlotter
   - One bin per house; x-axis label = "{index} ({total_consumption} kWh)".
   - For each (house, sizing criterion), computes mean metrics:
     NPV/kWh (lifetime), ROI, CAPEX/kWh, self-consumption,
     payback period (years), annual profit/kWh (€/kWh/yr). Layout: 2x3.
   - Plots grouped bar charts: one panel per metric, showing how each
     criterion (NEEG, NPV, ROI) performs in each consumption band.
   - Use case: "What is the impact of each sizing criterion for each
     consumption level, in aggregate over all houses?"

2) CriterionDistributionPlotter
   - For each house, shows the distribution of NPV, CAPEX, ROI, payback
     period and self-consumption across the cost space, broken down by
     optimization criterion (NEEG, NPV, ROI). One figure per house;
     violin or box plots per metric.
   - Use case: "For this house, how do the outcomes vary by criterion
     over the cost space?"

3) ROIVsNEEGPlotter
   - Compares sizing by maximising ROI vs minimising NEEG. For each house
     and each metric (ROI, CAPEX, NPV, payback period, self-consumption),
     shows the fraction of cost cells where the ROI-optimal outcome beats
     the NEEG-optimal outcome. One figure, five panels (one per metric).
   - Use case: "When does maximising ROI beat minimising NEEG, per house
     and per metric?"

4) CostSpaceDominancePlotter
   - For each cost cell (initial_cost_per_kW, cost_per_kW,
     feed_in_price_per_kW), determines which criterion *wins* (maximises
     the chosen metric, e.g. npv_per_kwh or roi).
   - Plots a 2D heatmap (cost_per_kW vs feed_in_price_per_kW, one slice of
     initial_cost_per_kW): each cell is coloured by the winning criterion.
   - Can restrict to one house_id or aggregate over all houses (mode).
   - Use case: "Across the cost space, which criterion gives the best
     outcome (e.g. best NPV per kWh) at each price point?"
"""

from dataclasses import dataclass
import os
from typing import Literal, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from source.pv.creation import PVSizingFilePathBuilder


def _load_pv_stats_for_merge(
    builder: PVSizingFilePathBuilder,
) -> pd.DataFrame | None:
    """
    Load pv_statistics.csv and return a dataframe with house_id,
    panel_type, mount_type, number_of_panels, self_sufficiency
    (and total_consumption_kwh) for merging with economical-space data.
    """
    path = builder.get_pv_statistics_csv_path()
    if not os.path.exists(path):
        return None
    pv = pd.read_csv(path)
    for col in ("self_sufficiency", "house_id", "panel_type", "mount_type"):
        if col not in pv.columns:
            return None
    out = pv.rename(columns={"n_panels": "number_of_panels"})
    cols = ["house_id", "panel_type", "mount_type", "number_of_panels",
            "self_sufficiency"]
    if "total_consumption_kwh" in pv.columns:
        cols.append("total_consumption_kwh")
    key: tuple[str, ...] = (
        "house_id", "panel_type", "mount_type", "number_of_panels"
    )
    return out[cols].drop_duplicates(subset=key)  # type: ignore[arg-type]


def _house_index_map(
    df: pd.DataFrame,
    house_col: str = "house_id",
    consumption_col: str | None = "total_consumption_kwh",
) -> pd.Series:
    """
    Map each house to a 1-based index (1, 2, 3, ...). Order: by
    total_consumption_kwh ascending, then by house_id. Used so all
    plots show the same short index in labels (e.g. "1 (4500 kWh)").
    """
    one_per_house = df[[house_col]].drop_duplicates()
    if consumption_col and consumption_col in df.columns:
        one_per_house = (
            df[[house_col, consumption_col]]
            .drop_duplicates(subset=[house_col])  # type: ignore[arg-type]
            .sort_values([consumption_col, house_col])
        )
    else:
        one_per_house = one_per_house.sort_values(
            [house_col])  # type: ignore[arg-type]
    one_per_house = one_per_house.reset_index(drop=True)
    one_per_house["house_index"] = np.arange(
        1, len(one_per_house) + 1, dtype=int)
    return cast(pd.Series, one_per_house.set_index(house_col)["house_index"])


def _load_consumption_per_house(
    builder: PVSizingFilePathBuilder,
) -> pd.DataFrame | None:
    """One row per house_id with total_consumption_kwh from pv_statistics."""
    path = builder.get_pv_statistics_csv_path()
    if not os.path.exists(path):
        return None
    pv = pd.read_csv(path)
    if "total_consumption_kwh" not in pv.columns:
        return None
    return cast(
        pd.DataFrame,
        pv[["house_id", "total_consumption_kwh"]]
        .groupby("house_id", as_index=False)
        .first(),
    )


# --- Overall statistics by consumption ---------------------------------------


@dataclass
class ConsumptionStatsPlotterConfig:
    """
    Configuration for ConsumptionStatsPlotter.

    Attributes:
        n_consumption_bins: Unused; each house is one bin with label
            "{index} ({total_consumption_kwh} kWh)".
        save_to_pdf: If True, save figure to results/sizing/; else show.
    """

    n_consumption_bins: int = 3  # kept for API; one bin per house is used
    save_to_pdf: bool = True


class ConsumptionStatsPlotter:
    """
    Overall statistics by consumption: impact of each criterion for all houses.

    Loads the three economical-space result files (one per sizing criterion:
    NEEG, NPV, ROI). Each row is a (house, cost triple) with the optimal
    PV config and metrics when sizing with that criterion. Each house has
    one total_consumption_kwh (from pv_statistics.csv). Each
    house is one bin with label "House {id}: {total_consumption_kwh} kWh".
    For each (house_label, criterion), computes the mean of:
    - npv_per_kwh (lifetime), roi, capex_per_kwh, self_consumption,
      payback_period (years), annual_profit_per_kwh (€/kWh per year).
    Plots grouped bar charts: one panel per metric, x = house (with
    total consumption in label), hue = criterion. Answers: "For each house,
    how does
    each sizing criterion perform on average?"
    """

    CRITERION_CONFIGS = [
        ("neeg", "NEEG"),
        ("npv", "NPV"),
        ("roi", "ROI"),
    ]

    METRIC_LABELS = {
        "npv_per_kwh": "Mean NPV / consumption (€/kWh, lifetime)",
        "roi": "Mean ROI",
        "capex_per_kwh": "Mean CAPEX / consumption (€/kWh)",
        "self_consumption": "Mean self-consumption",
        "self_sufficiency": "Mean self-sufficiency",
        "payback_period": "Mean payback period (years)",
        "annual_profit_per_kwh": "Mean annual profit / consumption (€/kWh/yr)",
    }

    def __init__(self, config: ConsumptionStatsPlotterConfig):
        """Store config and the PV sizing path builder."""
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self) -> None:
        """Load data, draw grouped bar charts, save or show."""
        df_agg = self._load_and_aggregate()
        if df_agg is None or df_agg.empty:
            print("No data for consumption stats plot.")
            return

        fig, axes = self._plot_impact_by_consumption(df_agg)
        self._finalize_stats_plot(fig, axes)

    def _load_consumption_per_house(self) -> pd.DataFrame | None:
        """
        Load one row per house_id with total_consumption_kwh from
        pv_statistics.csv. Returns None if file or column missing.
        """
        return _load_consumption_per_house(self._builder)

    def _load_and_aggregate(self) -> pd.DataFrame | None:
        """
        Load all three economical-space CSVs, merge with consumption (one
        total_consumption_kwh per house). Each house gets one label
        "{index} ({total_consumption_kwh} kWh)". Add npv_per_kwh,
        capex_per_kwh, annual_profit_per_kwh; aggregate mean of the six
        metrics (incl. payback_period in years) per (house_label, criterion).
        Returns None if no data.
        """
        dfs: list[pd.DataFrame] = []
        for indicator_name, label in self.CRITERION_CONFIGS:
            csv_path = self._builder.get_economical_space_csv_path(
                indicator_name
            )
            if not os.path.exists(csv_path):
                continue
            df_ind = pd.read_csv(csv_path)
            df_ind["criterion"] = label
            dfs.append(df_ind)

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        # Attach self_sufficiency from pv_statistics (per optimal config).
        stats_keys = ["house_id", "panel_type",
                      "mount_type", "number_of_panels"]
        pv_stats = _load_pv_stats_for_merge(self._builder)
        if pv_stats is not None and all(k in df.columns for k in stats_keys):
            df = df.merge(
                pv_stats[stats_keys + ["self_sufficiency"]],
                on=stats_keys,
                how="left",
            )
        consumption = _load_consumption_per_house(self._builder)
        if consumption is None:
            return None

        # One bin per house: label = "{index} ({total_consumption_kwh} kWh)".
        idx_ser = _house_index_map(consumption)
        consumption["house_index"] = consumption["house_id"].map(
            idx_ser)  # type: ignore[arg-type]
        consumption["house_label"] = (
            consumption["house_index"].astype(str) + " ("
            + consumption["total_consumption_kwh"].round(0).astype(int)
            .astype(str) + " kWh)"
        )

        df = df.merge(
            consumption[["house_id", "total_consumption_kwh", "house_label"]],
            on="house_id",
            how="left",
        )
        # Normalise NPV and CAPEX by house consumption for fair comparison.
        df["npv_per_kwh"] = np.where(
            df["total_consumption_kwh"] > 0,
            df["npv"] / df["total_consumption_kwh"],
            np.nan,
        )
        df["capex_per_kwh"] = np.where(
            df["total_consumption_kwh"] > 0,
            df["capex"] / df["total_consumption_kwh"],
            np.nan,
        )
        df["annual_profit_per_kwh"] = np.where(
            df["total_consumption_kwh"] > 0,
            df["annual_profit"] / df["total_consumption_kwh"],
            np.nan,
        )
        # One row per (house_label, criterion) with mean metrics; keep
        # total_consumption_kwh for x-axis order (by consumption).
        agg_cols = [
            "npv_per_kwh", "roi", "capex_per_kwh", "self_consumption",
            "self_sufficiency", "payback_period", "annual_profit_per_kwh",
        ]
        if "self_sufficiency" not in df.columns:
            agg_cols = [c for c in agg_cols if c != "self_sufficiency"]
        mean_df = df.groupby(
            ["house_label", "criterion"],
            dropna=False,
        ).agg(
            {c: "mean" for c in agg_cols}
            | {"total_consumption_kwh": "first"}
        ).reset_index()
        return cast(pd.DataFrame, mean_df)

    def _plot_impact_by_consumption(
        self,
        df_agg: pd.DataFrame,
    ) -> tuple[Figure, np.ndarray]:
        """
        Draw one grouped bar chart per metric: x = house_label
        (House id: X kWh), hue = criterion (NEEG, NPV, ROI), y = mean value.
        Layout: 2 rows x 4 columns (7 metrics + 1 empty).
        """
        metrics = [
            "npv_per_kwh", "roi", "capex_per_kwh", "self_consumption",
            "self_sufficiency", "payback_period", "annual_profit_per_kwh",
        ]
        metrics = [m for m in metrics if m in df_agg.columns]
        n_metrics = len(metrics)
        # Order x-axis by total consumption (low to high).
        x_order = (
            df_agg.sort_values("total_consumption_kwh")
            .drop_duplicates("house_label")["house_label"]
            .tolist()
        )
        # 2x4 subplot matrix (7 panels used).
        n_cols = 4
        n_rows = 2
        fig, axes_2d = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes = axes_2d.flat

        for ax, metric in zip(axes, metrics):
            sns.barplot(
                data=df_agg,
                x="house_label",
                y=metric,
                hue="criterion",
                hue_order=["NEEG", "NPV", "ROI"],
                palette=["#2ecc71", "#3498db", "#e74c3c"],
                ax=ax,
                errorbar=None,
                order=x_order,
            )
            ax.set_title(self.METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("House (total consumption)")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend(title="Sizing criterion")
            ax.tick_params(axis="x", rotation=90)
        if n_metrics < len(axes):
            axes[n_metrics].set_visible(False)

        fig.suptitle(
            "Impact of sizing criterion by consumption level (all houses)",
            y=1.02,
        )
        return fig, axes_2d

    def _finalize_stats_plot(
        self,
        fig: Figure,
        axes: np.ndarray,
    ) -> None:
        """Save to results/sizing/cost_space_consumption_stats.pdf or show."""
        plt.tight_layout()
        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            file_name = "cost_space_consumption_stats.pdf"
            file_path = os.path.join(folder, file_name)
            fig.savefig(file_path)
            plt.close(fig)
        else:
            plt.show()


# --- Criterion distribution per house ----------------------------------------


@dataclass
class CriterionDistributionPlotterConfig:
    """
    Configuration for CriterionDistributionPlotter.

    Attributes:
        house_ids: Houses to plot; one figure per house. If None, plot
            all houses present in the economical-space CSVs.
        plot_type: "violin" or "box" for distribution per criterion.
        save_to_pdf: If True, save to results/sizing/; else show.
    """

    house_ids: list[int] | None = None
    plot_type: Literal["violin", "box"] = "violin"
    save_to_pdf: bool = True


class CriterionDistributionPlotter:
    """
    Distribution of NPV, CAPEX, ROI, payback period and self-consumption
    by optimization criterion, for each house.

    Loads the three economical-space CSVs (one row per house and cost
    triple per criterion). For each house, draws one figure with five
    panels: each panel shows the distribution of one metric (over the
    cost space) with x = criterion (NEEG, NPV, ROI). Use violin or box
    plots to compare how each sizing criterion leads to different
    outcome distributions.
    """

    CRITERION_CONFIGS = [
        ("neeg", "NEEG"),
        ("npv", "NPV"),
        ("roi", "ROI"),
    ]

    METRICS = [
        "npv", "capex", "roi", "payback_period", "self_consumption",
        "self_sufficiency",
    ]
    METRIC_LABELS = {
        "npv": "NPV (€)",
        "capex": "CAPEX (€)",
        "roi": "ROI",
        "payback_period": "Payback period (years)",
        "self_consumption": "Self-consumption",
        "self_sufficiency": "Self-sufficiency",
    }

    def __init__(self, config: CriterionDistributionPlotterConfig):
        """Store config and the PV sizing path builder."""
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self) -> None:
        """Load data, then one figure per house with distribution plots."""
        df = self._load_data()
        if df is None or df.empty:
            print("No data for criterion distribution plot.")
            return

        house_ids = self._config.house_ids
        if house_ids is None:
            house_ids = sorted(df["house_id"].unique().tolist())

        consumption = _load_consumption_per_house(self._builder)
        if consumption is not None:
            index_map = _house_index_map(consumption)
        else:
            index_map = _house_index_map(
                cast(pd.DataFrame, df[["house_id"]].drop_duplicates()),
                consumption_col=None,
            )

        for house_id in house_ids:
            df_house = cast(
                pd.DataFrame,
                df[df["house_id"] == house_id],
            )
            if df_house.empty:
                continue
            house_index = int(index_map.loc[house_id])
            fig, axes_2d = self._plot_one_house(
                df_house, house_id, house_index=house_index
            )
            self._finalize_plot(fig, house_id)

    def _load_data(self) -> pd.DataFrame | None:
        """
        Load the three economical-space CSVs, add criterion column,
        return concatenated dataframe. Columns include house_id,
        criterion, npv, capex, roi, payback_period, self_consumption.
        """
        dfs: list[pd.DataFrame] = []
        for indicator_name, label in self.CRITERION_CONFIGS:
            csv_path = self._builder.get_economical_space_csv_path(
                indicator_name
            )
            if not os.path.exists(csv_path):
                continue
            df_ind = pd.read_csv(csv_path)
            df_ind["criterion"] = label
            dfs.append(df_ind)

        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        # Attach self_sufficiency from pv_statistics (per optimal config).
        stats_keys = ["house_id", "panel_type",
                      "mount_type", "number_of_panels"]
        pv_stats = _load_pv_stats_for_merge(self._builder)
        if pv_stats is not None and all(k in df.columns for k in stats_keys):
            df = df.merge(
                pv_stats[stats_keys + ["self_sufficiency"]],
                on=stats_keys,
                how="left",
            )
        return df

    def _plot_one_house(
        self,
        df_house: pd.DataFrame,
        house_id: int,
        house_index: int | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """
        One figure per house: 2x3 layout, six panels for the six
        metrics. In each panel, distribution (violin or box) by criterion.
        """
        fig, axes_2d = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes_2d.flat
        metrics = [m for m in self.METRICS if m in df_house.columns]
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            if self._config.plot_type == "violin":
                sns.violinplot(
                    data=df_house,
                    x="criterion",
                    y=metric,
                    hue="criterion",
                    order=["NEEG", "NPV", "ROI"],
                    hue_order=["NEEG", "NPV", "ROI"],
                    palette=["#2ecc71", "#3498db", "#e74c3c"],
                    legend=False,
                    ax=ax,
                )
            else:
                sns.boxplot(
                    data=df_house,
                    x="criterion",
                    y=metric,
                    hue="criterion",
                    order=["NEEG", "NPV", "ROI"],
                    hue_order=["NEEG", "NPV", "ROI"],
                    palette=["#2ecc71", "#3498db", "#e74c3c"],
                    legend=False,
                    ax=ax,
                )
            ax.set_title(self.METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("Optimization criterion")
            ax.set_ylabel("")
        if len(metrics) < len(axes):
            axes[len(metrics)].set_visible(False)
        title_idx = house_index if house_index is not None else house_id
        fig.suptitle(
            f"Distribution of outcomes by criterion ({title_idx})",
            y=1.02,
        )
        return fig, axes_2d

    def _finalize_plot(self, fig: Figure, house_id: int) -> None:
        """Save to results/sizing/criterion_distribution_{house_id}.pdf."""
        plt.tight_layout()
        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            ext = "pdf"
            file_name = f"criterion_distribution_{house_id}.{ext}"
            file_path = os.path.join(folder, file_name)
            fig.savefig(file_path)
            plt.close(fig)
        else:
            plt.show()


# --- ROI vs NEEG comparison -----------------------------------------------


@dataclass
class ROIVsNEEGPlotterConfig:
    """
    Configuration for ROIVsNEEGPlotter.

    Attributes:
        house_ids: Houses to include. If None, use all houses in the
            merged ROI and NEEG data.
        save_to_pdf: If True, save to results/sizing/; else show.
    """

    house_ids: list[int] | None = None
    save_to_pdf: bool = True


class ROIVsNEEGPlotter:
    """
    Compare maximising ROI vs minimising NEEG: for each house and each
    metric, show the fraction of cost cells where the ROI-optimal outcome
    beats the NEEG-optimal outcome.

    Loads economical_space_roi.csv and economical_space_neeg.csv, merges
    on (house_id, cost triple). For each metric (ROI, CAPEX, NPV,
    payback period, self-consumption), "ROI beats NEEG" means: higher
    is better (ROI, NPV, self-consumption) or lower is better (CAPEX,
    payback). One figure with five panels; each panel shows per-house
    fraction of cost cells where ROI beats NEEG on that metric.
    """

    COST_KEYS = [
        "house_id",
        "initial_cost_per_kW",
        "cost_per_kW",
        "feed_in_price_per_kW",
    ]
    METRICS = [
        "roi", "capex", "npv", "payback_period", "self_consumption",
        "self_sufficiency",
    ]
    # Labels clarify: y-axis is "Fraction where ROI beats NEEG".
    # For payback, "ROI wins" = ROI's payback is lower (shorter) than NEEG's.
    METRIC_LABELS = {
        "roi": "ROI (ROI wins if higher)",
        "capex": "CAPEX (ROI wins if lower)",
        "npv": "NPV (ROI wins if higher)",
        "payback_period": "Payback (ROI wins if shorter)",
        "self_consumption": "Self-consumption (ROI wins if higher)",
        "self_sufficiency": "Self-sufficiency (ROI wins if higher)",
    }
    # ROI beats NEEG: higher is better (roi_val > neeg_val) or lower better
    # (roi_val < neeg_val for capex, payback).
    _HIGHER_BETTER = {"roi", "npv", "self_consumption", "self_sufficiency"}
    _LOWER_BETTER = {"capex", "payback_period"}

    def __init__(self, config: ROIVsNEEGPlotterConfig):
        """Store config and the PV sizing path builder."""
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self) -> None:
        """Load ROI and NEEG data, compute win fractions, draw and save."""
        df_agg = self._load_and_aggregate()
        if df_agg is None or df_agg.empty:
            print("No data for ROI vs NEEG comparison plot.")
            return

        fig, axes_2d = self._plot_fractions(df_agg)
        self._finalize_plot(fig)

    def _load_consumption_per_house(self) -> pd.DataFrame | None:
        """One row per house_id with total_consumption_kwh (pv_statistics)."""
        return _load_consumption_per_house(self._builder)

    def _load_and_aggregate(self) -> pd.DataFrame | None:
        """
        Load ROI and NEEG CSVs, merge on cost keys, compute for each
        (house_id, metric) the fraction of cost cells where ROI beats
        NEEG. Return long-format: house_id, house_label, metric, fraction.
        """
        path_roi = self._builder.get_economical_space_csv_path("roi")
        path_neeg = self._builder.get_economical_space_csv_path("neeg")
        if not os.path.exists(path_roi) or not os.path.exists(path_neeg):
            return None

        df_roi = pd.read_csv(path_roi)
        df_neeg = pd.read_csv(path_neeg)
        common = [
            c for c in self.COST_KEYS
            if c in df_roi.columns and c in df_neeg.columns
        ]
        if len(common) < len(self.COST_KEYS):
            return None

        # Attach self_sufficiency from pv_statistics (per optimal config).
        stats_keys = ["house_id", "panel_type",
                      "mount_type", "number_of_panels"]
        pv_stats = _load_pv_stats_for_merge(self._builder)
        if pv_stats is not None and all(
                k in df_roi.columns for k in stats_keys):
            df_roi = df_roi.merge(
                pv_stats[stats_keys + ["self_sufficiency"]],
                on=stats_keys,
                how="left",
            )
            df_neeg = df_neeg.merge(
                pv_stats[stats_keys + ["self_sufficiency"]],
                on=stats_keys,
                how="left",
            )

        merged = df_roi.merge(
            df_neeg,
            on=common,
            suffixes=("_roi", "_neeg"),
        )
        if merged.empty:
            return None

        house_ids = self._config.house_ids
        if house_ids is not None:
            merged = merged[merged["house_id"].isin(house_ids)]
            if merged.empty:
                return None

        # ROI beats NEEG: for higher-better metrics roi > neeg;
        # for lower-better metrics roi < neeg.
        records: list[dict] = []
        merged_df = cast(pd.DataFrame, merged)
        for house_id in merged_df["house_id"].unique():
            df_h = cast(
                pd.DataFrame, merged_df[merged_df["house_id"] == house_id])
            for metric in self.METRICS:
                roi_col = f"{metric}_roi"
                neeg_col = f"{metric}_neeg"
                if roi_col not in df_h.columns or neeg_col not in df_h.columns:
                    continue
                if metric in self._HIGHER_BETTER:
                    wins = (df_h[roi_col] > df_h[neeg_col]).astype(float)
                else:
                    wins = (df_h[roi_col] < df_h[neeg_col]).astype(float)
                fraction = wins.mean()
                records.append({
                    "house_id": house_id,
                    "metric": metric,
                    "fraction": fraction,
                })

        if not records:
            return None
        df_agg = pd.DataFrame(records)
        # House label for x-axis: "{index} ({total_consumption_kwh} kWh)".
        consumption = self._load_consumption_per_house()
        if consumption is not None:
            idx_ser = _house_index_map(consumption)
            consumption["house_index"] = consumption["house_id"].map(
                idx_ser)  # type: ignore[arg-type]
            df_agg = df_agg.merge(
                consumption[
                    ["house_id", "total_consumption_kwh", "house_index"]
                ],
                on="house_id",
                how="left",
            )
            cons = df_agg["total_consumption_kwh"].round(0).fillna(0)
            df_agg["house_label"] = (
                df_agg["house_index"].astype(str) + " ("
                + cons.astype(int).astype(str) + " kWh)"
            )
        else:
            idx_ser = _house_index_map(
                cast(pd.DataFrame, df_agg[["house_id"]].drop_duplicates()),
                consumption_col=None,
            )
            df_agg["house_label"] = (
                df_agg["house_id"].map(idx_ser).astype(str)  # type: ignore
            )
        return cast(pd.DataFrame, df_agg)

    def _plot_fractions(
        self,
        df_agg: pd.DataFrame,
    ) -> tuple[Figure, np.ndarray]:
        """
        One panel per metric: x = house, y = fraction where ROI beats NEEG.
        """
        fig, axes_2d = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes_2d.flat
        # Sort houses by total consumption (ascending); fallback to house_id.
        dup = df_agg.drop_duplicates("house_id")
        if "total_consumption_kwh" in df_agg.columns:
            house_order = (
                dup.sort_values("total_consumption_kwh")[
                    "house_label"].tolist()
            )
        else:
            house_order = dup.sort_values("house_id")["house_label"].tolist()

        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx]
            df_m = cast(
                pd.DataFrame,
                df_agg[df_agg["metric"] == metric],
            )
            sns.barplot(
                data=df_m,
                x="house_label",
                y="fraction",
                order=house_order,
                color="#3498db",
                ax=ax,
            )
            ax.set_title(self.METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("House")
            ax.set_ylabel("Fraction cost cells\n(ROI beats NEEG)")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=90)
        fig.suptitle(
            "When maximising ROI beats minimising NEEG (all houses)",
            y=1.02,
        )
        return fig, axes_2d

    def _finalize_plot(self, fig: Figure) -> None:
        """Save to results/sizing/roi_vs_neeg.pdf or show."""
        plt.tight_layout()
        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            file_path = os.path.join(folder, "roi_vs_neeg.pdf")
            fig.savefig(file_path)
            plt.close(fig)
        else:
            plt.show()


# --- Cost-space dominance (heatmap) ------------------------------------------


@dataclass
class CostSpaceDominancePlotterConfig:
    """
    Configuration for CostSpaceDominancePlotter.

    Attributes:
        house_id: If set, restrict to this house; if None, use all houses
            and at each cost cell the winner is the mode across houses.
        win_metric: Metric used to pick the winner at each cost cell:
            npv_per_kwh, roi, or neeg_per_kwh (all relative to consumption
            where applicable).
        initial_cost_per_kW: Fix this dimension for the 2D heatmap (one
            slice of the cost space). If None, use first value in data.
        save_to_pdf: If True, save to results/sizing/; else show.
    """

    house_id: int | None = None
    win_metric: Literal["npv_per_kwh", "roi", "neeg_per_kwh"] = "npv_per_kwh"
    initial_cost_per_kW: float | None = None
    save_to_pdf: bool = True


class CostSpaceDominancePlotter:
    """
    Cost-space dominance heatmap: which criterion wins at each cost cell.

    For each (house, initial_cost_per_kW, cost_per_kW, feed_in_price_per_kW),
    the three economical-space files give the outcome when sizing with NEEG,
    NPV, or ROI. This plotter picks the criterion that maximises the chosen
    win_metric (e.g. npv_per_kwh) at that cell. It then plots a 2D heatmap:
    axes = cost_per_kW and feed_in_price_per_kW (one slice of
    initial_cost_per_kW); colour = winning criterion (NEEG / NPV / ROI).
    If house_id is None, multiple houses are aggregated: at each (cost_per_kW,
    feed_in_price_per_kW) the winner is the mode across houses. Answers:
    "At each price point in the cost space, which sizing criterion gives
    the best result?"
    """

    CRITERION_CONFIGS = [
        ("neeg", "NEEG"),
        ("npv", "NPV"),
        ("roi", "ROI"),
    ]

    def __init__(self, config: CostSpaceDominancePlotterConfig):
        """Store config and the PV sizing path builder."""
        self._config = config
        self._builder = PVSizingFilePathBuilder()

    def plot(self) -> None:
        """Load winners per cost cell, draw heatmap, save or show."""
        df_win = self._load_data()
        if df_win is None or df_win.empty:
            print("No data available for cost-space dominance plot.")
            return

        fig, ax = self._plot_heatmap(df_win)
        self._finalize_plot(fig, ax)

    def _load_data(self) -> pd.DataFrame | None:
        """
        Load the three economical-space CSVs, merge with consumption, compute
        npv_per_kwh and neeg_per_kwh. For each cost cell, select the row
        (criterion) that maximises win_metric. Return a dataframe with one
        row per (house_id?, ick, ck, pik) and column criterion = winner.
        """
        dfs: list[pd.DataFrame] = []
        for indicator_name, label in self.CRITERION_CONFIGS:
            csv_path = self._builder.get_economical_space_csv_path(
                indicator_name
            )
            if not os.path.exists(csv_path):
                continue
            df_ind = pd.read_csv(csv_path)
            df_ind["criterion"] = label
            dfs.append(df_ind)

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)

        # Optionally restrict to a single house.
        if self._config.house_id is not None:
            df = df[df["house_id"] == self._config.house_id].copy()
            if df.empty:
                return None

        consumption = self._load_consumption_per_house()
        if consumption is None:
            return None

        df = df.merge(
            consumption,
            on="house_id",
            how="left",
        )
        # Normalise by consumption for consumption-relative metrics.
        df["npv_per_kwh"] = np.where(
            df["total_consumption_kwh"] > 0,
            df["npv"] / df["total_consumption_kwh"],
            np.nan,
        )
        df["neeg_per_kwh"] = np.where(
            df["total_consumption_kwh"] > 0,
            df["neeg"] / df["total_consumption_kwh"],
            np.nan,
        )

        # At each cost cell, winner = criterion with max win_metric.
        win_col = self._config.win_metric
        if win_col == "npv_per_kwh" and "npv_per_kwh" not in df.columns:
            win_col = "npv"
        elif win_col == "neeg_per_kwh" and "neeg_per_kwh" not in df.columns:
            win_col = "neeg"

        df_win = (
            df.loc[df.groupby(
                ["house_id", "initial_cost_per_kW", "cost_per_kW",
                 "feed_in_price_per_kW"],
                dropna=False,
            )[win_col].idxmax()]
            .reset_index(drop=True)[
                ["house_id", "initial_cost_per_kW", "cost_per_kW",
                 "feed_in_price_per_kW", "criterion"]
            ]
        )

        if self._config.house_id is not None:
            df_win = df_win.drop(columns=["house_id"])

        return df_win

    def _load_consumption_per_house(self) -> pd.DataFrame | None:
        """
        Load one row per house_id with total_consumption_kwh from
        pv_statistics.csv. Returns None if file or column missing.
        """
        csv_path = self._builder.get_pv_statistics_csv_path()
        if not os.path.exists(csv_path):
            return None
        pv = pd.read_csv(csv_path)
        if "total_consumption_kwh" not in pv.columns:
            return None
        consumption = cast(
            pd.DataFrame,
            pv[["house_id", "total_consumption_kwh"]]
            .groupby("house_id", as_index=False)
            .first(),
        )
        return consumption

    def _plot_heatmap(self, df_win: pd.DataFrame) -> tuple[Figure, Axes]:
        """
        Restrict to one initial_cost_per_kW slice, aggregate winner per
        (cost_per_kW, feed_in_price_per_kW) if multiple houses, pivot to
        matrix and draw discrete heatmap (colour = NEEG / NPV / ROI).
        """
        # Restrict to a single slice of the cost space for 2D heatmap.
        ick = self._config.initial_cost_per_kW
        if ick is not None:
            df_win = cast(
                pd.DataFrame,
                df_win[np.isclose(df_win["initial_cost_per_kW"], ick)].copy(),
            )
        elif "initial_cost_per_kW" in df_win.columns:
            ick = df_win["initial_cost_per_kW"].iloc[0]
            df_win = cast(
                pd.DataFrame,
                df_win[np.isclose(df_win["initial_cost_per_kW"], ick)].copy(),
            )

        if "house_id" in df_win.columns:
            if df_win["house_id"].nunique() > 1:
                def _mode_criterion(s: pd.Series) -> str:
                    m = s.mode()
                    return m.iloc[0] if len(m) else s.iloc[0]

                winner = (
                    df_win.groupby(
                        ["cost_per_kW", "feed_in_price_per_kW"],
                        dropna=False,
                    )["criterion"]
                    .agg(_mode_criterion)
                    .reset_index()
                )
            else:
                winner = df_win[
                    ["cost_per_kW", "feed_in_price_per_kW", "criterion"]
                ].drop_duplicates()
        else:
            winner = df_win[
                ["cost_per_kW", "feed_in_price_per_kW", "criterion"]
            ].copy()

        # One row per (cost_per_kW, feed_in_price_per_kW): winning criterion.
        pivot = winner.pivot_table(
            index="feed_in_price_per_kW",
            columns="cost_per_kW",
            values="criterion",
            aggfunc="first",
        )
        # Map criterion names to 0/1/2 for discrete colouring.
        criterion_order = ["NEEG", "NPV", "ROI"]
        cmap_colors = ["#2ecc71", "#3498db", "#e74c3c"]
        cat_to_num = {c: i for i, c in enumerate(criterion_order)}
        pivot_ord = cast(
            pd.DataFrame,
            pivot.reindex(
                index=pivot.index.sort_values(ascending=False),
                columns=pivot.columns.sort_values(),
            ),
        )
        pivot_num = cast(
            pd.DataFrame,
            pivot_ord.apply(lambda s: s.map(cat_to_num)).astype(float),
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        discrete_cmap = ListedColormap(cmap_colors)
        sns.heatmap(
            pivot_num,
            ax=ax,
            cmap=discrete_cmap,
            vmin=0,
            vmax=2,
            cbar_kws={
                "ticks": [1 / 3, 1.0, 5 / 3],
                "label": "Criterion",
            },
        )
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_ticklabels(criterion_order)
        ax.set_xlabel("cost_per_kW (€/kWh)")
        ax.set_ylabel("feed_in_price_per_kW (€/kWh)")
        ick_str = f"{ick:.0f}" if ick is not None else "varies"
        ax.set_title(
            f"Winning criterion by cost cell (win={self._config.win_metric}, "
            f"initial_cost_per_kW={ick_str})"
        )
        return fig, ax

    def _finalize_plot(
        self,
        fig: Figure,
        ax: Axes,
    ) -> None:
        """Save to results/sizing/cost_space_dominance_*.pdf or show."""
        plt.tight_layout()
        if self._config.save_to_pdf:
            folder = self._builder.get_pv_sizing_folder()
            house_suffix = (
                f"{self._config.house_id}"
                if self._config.house_id is not None
                else "all"
            )
            file_name = (
                f"cost_space_dominance_{house_suffix}_"
                f"{self._config.win_metric}.pdf"
            )
            file_path = os.path.join(folder, file_name)
            fig.savefig(file_path)
            plt.close(fig)
        else:
            plt.show()

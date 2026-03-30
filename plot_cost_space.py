"""
Run cost-space plots: consumption stats, criterion distribution,
ROI vs NEEG comparison, dominance heatmap.
Usage: python plot_cost_space.py
"""

from source.plot.cost_dominance import DominancePlotter, DominancePlotterConfig
from source.plot.cost_space import (
    ConsumptionStatsPlotter,
    ConsumptionStatsPlotterConfig,
    CostSpaceDominancePlotter,
    CostSpaceDominancePlotterConfig,
    CriterionDistributionPlotter,
    CriterionDistributionPlotterConfig,
    ROIVsNEEGPlotter,
    ROIVsNEEGPlotterConfig,
)

if __name__ == "__main__":
    # Overall statistics for all houses:
    # impact of each criterion by consumption
    stats_config = ConsumptionStatsPlotterConfig(
        n_consumption_bins=3,
        save_to_pdf=True,
    )
    ConsumptionStatsPlotter(stats_config).plot()

    # Distribution of NPV, CAPEX, ROI, payback, self-consumption
    # by criterion, one figure per house (house_ids=None => all houses)
    dist_config = CriterionDistributionPlotterConfig(
        house_ids=None,
        plot_type="violin",
        save_to_pdf=True,
    )
    CriterionDistributionPlotter(dist_config).plot()

    # When does maximising ROI beat minimising NEEG? (all houses)
    roi_vs_neeg_config = ROIVsNEEGPlotterConfig(
        house_ids=None,
        save_to_pdf=True,
    )
    ROIVsNEEGPlotter(roi_vs_neeg_config).plot()

    # Dominance heatmap (optional: one house or all)
    dominance_config_high_initial_cost = CostSpaceDominancePlotterConfig(
        house_id=None,
        win_metric="roi",
        initial_cost_per_kW=1240.0,
        save_to_pdf=True,
    )
    dominance_config_low_initial_cost = CostSpaceDominancePlotterConfig(
        house_id=None,
        win_metric="roi",
        initial_cost_per_kW=750.0,
        save_to_pdf=True,
    )
    for dominance_config in [dominance_config_high_initial_cost,
                             dominance_config_low_initial_cost]:
        CostSpaceDominancePlotter(dominance_config).plot()

    # Cost dominance plot: one figure per house
    cost_dominance_config = DominancePlotterConfig(
        house_ids=None,
        base_indicator="neeg",
        base_label="NEEG",
        comparison_indicator="roi",
        comparison_label="ROI",
        save_to_pdf=True,
    )
    DominancePlotter(cost_dominance_config).plot()

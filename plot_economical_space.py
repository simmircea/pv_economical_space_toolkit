from source.plot.economical_space import (
    Dominance3DPlotter,
    DominancePlotter, DominancePlotterConfig, ViolinPlotter,
    ViolinPlotterConfig
)

# python plot_economical_space.py


if __name__ == "__main__":
    house_id = 2000900
    config = ViolinPlotterConfig(
        indicator_configs=[("neeg", "NEEG"), ("roi", "ROI")],
        house_id=house_id,
        save_to_pdf=True
    )
    plotter = ViolinPlotter(config)
    plotter.plot()

    config = DominancePlotterConfig(
        base_indicator="neeg",
        base_label="NEEG",
        comparison_indicator="roi",
        comparison_label="ROI",
        house_id=house_id,
        save_to_pdf=True
    )
    plotter = DominancePlotter(config)
    plotter.plot()

    config = DominancePlotterConfig(
        base_indicator="neeg",
        base_label="NEEG",
        comparison_indicator="roi",
        comparison_label="ROI",
        house_id=house_id,
        save_to_pdf=True
    )
    plotter = Dominance3DPlotter(config)
    plotter.plot()

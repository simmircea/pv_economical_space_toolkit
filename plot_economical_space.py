from source.plot.economical_space import (
    ViolinPlotter, ViolinPlotterConfig)


if __name__ == "__main__":
    house_id = 2000900
    config = ViolinPlotterConfig(
        indicator_configs=[("neeg", "NEEG"), ("roi", "ROI")],
        house_id=house_id,
        save_to_pdf=True
    )
    plotter = ViolinPlotter(config)
    plotter.plot()

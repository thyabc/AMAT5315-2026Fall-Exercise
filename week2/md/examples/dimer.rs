use md::{Euler, VelocityVerlet, run_dimer};
use plotters::prelude::*;
use std::{error::Error, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let euler = run_dimer(&Euler, 0.01, 500);
    let verlet = run_dimer(&VelocityVerlet, 0.01, 500);
    let long_verlet = run_dimer(&VelocityVerlet, 0.01, 5000);
    let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("../dimer.png");
    let root = BitMapBackend::new(&output, (1400, 620)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.titled(
        "Isolated Lennard-Jones dimer | initial separation 1.2 | dt = 0.01",
        ("sans-serif", 25),
    )?;
    let panels = root.split_evenly((1, 2));

    let mut left = ChartBuilder::on(&panels[0])
        .caption("Both integrators: 500 steps", ("sans-serif", 23))
        .margin(24)
        .x_label_area_size(52)
        .y_label_area_size(75)
        .build_cartesian_2d(0.0..5.0, -0.1..2.1)?;
    left.configure_mesh()
        .x_desc("Time t")
        .y_desc("Relative energy error (E - E0) / |E0|")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 16))
        .light_line_style(RGBColor(235, 235, 235))
        .draw()?;
    let blue = RGBColor(35, 100, 180);
    let orange = RGBColor(210, 85, 20);
    left.draw_series(LineSeries::new(
        euler.iter().map(|s| (s.time, s.relative_error)),
        orange.stroke_width(3),
    ))?
    .label("Forward Euler")
    .legend(move |(x, y)| PathElement::new([(x, y), (x + 30, y)], orange.stroke_width(3)));
    left.draw_series(LineSeries::new(
        verlet.iter().map(|s| (s.time, s.relative_error)),
        blue.stroke_width(3),
    ))?
    .label("Velocity-Verlet")
    .legend(move |(x, y)| PathElement::new([(x, y), (x + 30, y)], blue.stroke_width(3)));
    left.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(RGBColor(200, 200, 200))
        .label_font(("sans-serif", 18))
        .draw()?;

    let mut right = ChartBuilder::on(&panels[1])
        .caption("Velocity-Verlet: 5000 steps", ("sans-serif", 23))
        .margin(24)
        .x_label_area_size(52)
        .y_label_area_size(75)
        .build_cartesian_2d(0.0..50.0, -0.5..0.5)?;
    right.configure_mesh()
        .x_desc("Time t")
        .y_desc("Relative energy error x 1000")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 16))
        .light_line_style(RGBColor(235, 235, 235))
        .draw()?;
    right.draw_series(LineSeries::new(
        long_verlet.iter().map(|s| (s.time, 1000.0 * s.relative_error)),
        blue.stroke_width(1),
    ))?;
    root.present()?;
    println!("Saved {}", output.display());
    Ok(())
}

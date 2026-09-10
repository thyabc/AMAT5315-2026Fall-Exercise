use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "md",
    version,
    about = "Two-dimensional Lennard-Jones molecular dynamics"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Record an equilibrated periodic fluid in reduced LJ units.
    Run(RunArgs),
    /// Validate a recording and check production energy, temperature, and speeds.
    Check {
        #[arg(long, default_value = "results/fluid")]
        input: PathBuf,
    },
    /// Animate recorded production frames as an MP4 video.
    Video {
        #[arg(long, default_value = "results/fluid")]
        input: PathBuf,
        #[arg(long, default_value = "results/fluid.mp4")]
        output: PathBuf,
        #[arg(long, default_value = "python3")]
        python: PathBuf,
        #[arg(long, default_value_t = 30)]
        fps: u32,
    },
    /// Write the original Part 3 dimer comparison CSV (also the no-argument default).
    Dimer,
}

#[derive(Args)]
struct RunArgs {
    #[arg(long, default_value = "results/fluid")]
    output: PathBuf,
    #[arg(long)]
    overwrite: bool,
    #[arg(long, default_value_t = 100)]
    particles: usize,
    #[arg(long, default_value_t = 0.8)]
    density: f64,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long, default_value_t = 0.005)]
    dt: f64,
    #[arg(long, default_value_t = 10000)]
    equilibration_steps: usize,
    #[arg(long, default_value_t = 10000)]
    steps: usize,
    #[arg(long, default_value_t = 50)]
    sample_every: usize,
    #[arg(long, default_value_t = 2.5)]
    cutoff: f64,
}

fn execute(cli: Cli) -> Result<bool, String> {
    match cli.command {
        None | Some(Command::Dimer) => {
            dimer();
            Ok(true)
        }
        Some(Command::Run(args)) => {
            let config = md::FluidConfig {
                particles: args.particles,
                density: args.density,
                temperature: args.temperature,
                seed: args.seed,
                dt: args.dt,
                equilibration_steps: args.equilibration_steps,
                production_steps: args.steps,
                sample_every: args.sample_every,
                cutoff: args.cutoff,
            };
            md::record_run(&config, &args.output, args.overwrite)?;
            println!(
                "Recorded {} particles, {} equilibration steps, {} production steps in {}",
                config.particles,
                config.equilibration_steps,
                config.production_steps,
                args.output.display()
            );
            Ok(true)
        }
        Some(Command::Check { input }) => {
            let run = md::read_run(&input)?;
            let config = &run.metadata.config;
            let report = md::assess(
                &run.observables,
                &run.speeds,
                config.particles,
                config.temperature,
            )?;
            println!("{report}");
            Ok(report.passed())
        }
        Some(Command::Video {
            input,
            output,
            python,
            fps,
        }) => {
            md::render_video(&input, &output, &python, fps)?;
            println!("Wrote {}", output.display());
            Ok(true)
        }
    }
}

fn main() -> std::process::ExitCode {
    match execute(Cli::parse()) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::from(1),
        Err(error) => {
            eprintln!("md: {error}");
            std::process::ExitCode::from(2)
        }
    }
}

fn dimer() {
    println!("integrator,step,time,kinetic,potential,total,relative_error");
    for (name, samples) in [
        ("euler", md::run_dimer(&md::Euler, 0.01, 500)),
        (
            "velocity-verlet",
            md::run_dimer(&md::VelocityVerlet, 0.01, 500),
        ),
    ] {
        for s in samples {
            println!(
                "{name},{},{},{},{},{},{:.16e}",
                s.step, s.time, s.kinetic, s.potential, s.total, s.relative_error
            );
        }
    }
}

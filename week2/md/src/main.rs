fn main() {
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

use anyhow::{bail, Result};
use serde::Serialize;


#[derive(Debug, Clone, Serialize)]
pub struct TreeverseAction {
    pub action: String,
    pub step: usize,
    pub saved_states: usize,
}


#[derive(Debug, Clone)]
pub struct TreeversePlan {
    pub actions: Vec<TreeverseAction>,
    pub forward_calls: usize,
    pub peak_saved_states: usize,
}


// ==========================================================
// Binomial coefficient
// ==========================================================

fn binomial(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);

    let mut result = 1u128;

    for i in 1..=k {
        result =
            result
            * (n - k + i) as u128
            / i as u128;
    }

    result
}


// Smallest tau such that C(tau + delta, delta) >= N
fn binomial_fit(
    nsteps: usize,
    delta: usize,
) -> usize {
    let mut tau = 1usize;

    while binomial(
        tau + delta,
        delta,
    ) < nsteps as u128
    {
        tau += 1;
    }

    tau
}


// ==========================================================
// Treeverse split
// ==========================================================

fn midpoint(
    delta: usize,
    tau: usize,
    sigma: usize,
    phi: usize,
) -> usize {
    let denom =
        delta + tau;

    if denom == 0 {
        return phi;
    }

    let numerator =
        delta * sigma
        + tau * phi;

    // ceil(numerator / denom)
    let mut kappa =
        (numerator + denom - 1)
        / denom;

    if kappa >= phi && delta > 0 {
        kappa =
            (phi - 1)
                .max(sigma + 1);
    }

    kappa
}


fn add_action(
    actions: &mut Vec<TreeverseAction>,
    action: &str,
    step: usize,
    saved_states: usize,
) {
    actions.push(
        TreeverseAction {
            action: action.to_string(),
            step,
            saved_states,
        }
    );
}


// ==========================================================
// Recursive Treeverse construction
// ==========================================================

fn recurse(
    mut delta: usize,
    mut tau: usize,

    beta: usize,
    sigma: usize,
    mut phi: usize,

    actions: &mut Vec<TreeverseAction>,

    saved_states: &mut usize,
    peak_saved_states: &mut usize,
) -> Result<()> {

    // ------------------------------------------------------
    // Reconstruct and save s_sigma from checkpoint s_beta
    // ------------------------------------------------------

    if sigma > beta {
        if delta == 0 {
            bail!(
                "Treeverse exhausted checkpoint slots"
            );
        }

        delta -= 1;

        // Restore s_beta into working state.
        add_action(
            actions,
            "restore",
            beta,
            *saved_states,
        );

        // Replay beta -> sigma.
        for step in beta..sigma {
            add_action(
                actions,
                "call",
                step,
                *saved_states,
            );
        }

        // Save reconstructed s_sigma.
        *saved_states += 1;

        *peak_saved_states =
            (*peak_saved_states)
                .max(*saved_states);

        add_action(
            actions,
            "store",
            sigma,
            *saved_states,
        );
    }
    else if sigma < beta {
        bail!(
            "Treeverse failure: sigma < beta"
        );
    }

    // ------------------------------------------------------
    // Recurse through suffixes
    // ------------------------------------------------------

    let mut kappa =
        midpoint(
            delta,
            tau,
            sigma,
            phi,
        );

    while tau > 0 && kappa < phi {

        recurse(
            delta,
            tau,

            sigma,
            kappa,
            phi,

            actions,

            saved_states,
            peak_saved_states,
        )?;

        tau -= 1;
        phi = kappa;

        kappa =
            midpoint(
                delta,
                tau,
                sigma,
                phi,
            );
    }

    // ------------------------------------------------------
    // Reverse this timestep
    // ------------------------------------------------------

    add_action(
        actions,
        "grad",
        sigma,
        *saved_states,
    );

    // ------------------------------------------------------
    // Free temporary checkpoint
    // ------------------------------------------------------

    if sigma > beta {

        *saved_states -= 1;

        add_action(
            actions,
            "fetch",
            sigma,
            *saved_states,
        );
    }

    Ok(())
}


// ==========================================================
// Public scheduler
// ==========================================================

pub fn make_treeverse_plan(
    nsteps: usize,
    checkpoints: usize,
) -> Result<TreeversePlan> {

    if nsteps == 0 {
        bail!(
            "number of timesteps must be positive"
        );
    }

    if checkpoints == 0 {
        bail!(
            "Treeverse requires at least one checkpoint slot"
        );
    }

    let tau =
        binomial_fit(
            nsteps,
            checkpoints,
        );

    // s0 exists permanently.
    let mut saved_states =
        1usize;

    let mut peak_saved_states =
        1usize;

    let mut actions =
        Vec::new();

    recurse(
        checkpoints,
        tau,

        0,
        0,
        nsteps,

        &mut actions,

        &mut saved_states,
        &mut peak_saved_states,
    )?;

    if saved_states != 1 {
        bail!(
            "Treeverse ended with {} states; expected only s0",
            saved_states
        );
    }

    let forward_calls =
        actions
            .iter()
            .filter(
                |a| a.action == "call"
            )
            .count();

    Ok(
        TreeversePlan {
            actions,
            forward_calls,
            peak_saved_states,
        }
    )
}

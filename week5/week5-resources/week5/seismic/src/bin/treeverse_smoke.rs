#[path = "../treeverse.rs"]
mod treeverse;

use treeverse::make_treeverse_plan;


fn main() {
    let cases = [
        (1usize, 28680usize, 2usize),
        (3usize, 1695usize, 4usize),
        (5usize, 990usize, 6usize),
        (10usize, 642usize, 11usize),
    ];

    for (
        budget,
        expected_calls,
        expected_peak,
    ) in cases {

        let plan =
            make_treeverse_plan(
                240,
                budget,
            )
            .unwrap();

        let grads:
            Vec<usize> =
            plan.actions
                .iter()
                .filter(
                    |a| a.action == "grad"
                )
                .map(
                    |a| a.step
                )
                .collect();

        let expected_grads:
            Vec<usize> =
            (0..240)
                .rev()
                .collect();

        println!(
            "budget {:2}: calls = {:5}, peak = {:2}, grads = {}",
            budget,
            plan.forward_calls,
            plan.peak_saved_states,
            grads.len()
        );

        assert_eq!(
            plan.forward_calls,
            expected_calls
        );

        assert_eq!(
            plan.peak_saved_states,
            expected_peak
        );

        assert_eq!(
            grads,
            expected_grads
        );
    }

    println!(
        "PASS: Treeverse scheduler matches reference."
    );
}

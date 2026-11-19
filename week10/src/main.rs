use nalgebra::DMatrix;
use plotters::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use std::convert::TryInto;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

const N: usize = 512;
const RANKS: [usize; 5] = [16, 64, 128, 256, 512];

// Save a matrix as row-major little-endian f64.
fn write_matrix<P: AsRef<Path>>(path: P, matrix: &DMatrix<f64>) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            writer.write_all(&matrix[(i, j)].to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

// Read a row-major little-endian f64 matrix.
fn read_matrix<P: AsRef<Path>>(
    path: P,
    rows: usize,
    cols: usize,
) -> Result<DMatrix<f64>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    if bytes.len() != rows * cols * 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Incorrect matrix file size",
        )
        .into());
    }

    let mut matrix = DMatrix::<f64>::zeros(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            let offset = (i * cols + j) * 8;
            let value_bytes: [u8; 8] = bytes[offset..offset + 8].try_into()?;

            matrix[(i, j)] = f64::from_le_bytes(value_bytes);
        }
    }

    Ok(matrix)
}

// Save singular values.
fn write_vector<P: AsRef<Path>>(path: P, values: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);

    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

// Read singular values independently from disk.
fn read_vector<P: AsRef<Path>>(path: P) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    if bytes.len() % 8 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Incorrect vector file size",
        )
        .into());
    }

    let values = bytes
        .chunks_exact(8)
        .map(|chunk| {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            f64::from_le_bytes(arr)
        })
        .collect();

    Ok(values)
}

// Frobenius norm.
fn frobenius_norm(matrix: &DMatrix<f64>) -> f64 {
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// Relative Frobenius reconstruction error.
fn relative_error(original: &DMatrix<f64>, reconstructed: &DMatrix<f64>) -> f64 {
    let difference = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    difference / frobenius_norm(original)
}

// Reconstruct a matrix using only the first chi singular values.
fn reconstruct(
    u: &DMatrix<f64>,
    singular_values: &[f64],
    vt: &DMatrix<f64>,
    chi: usize,
) -> DMatrix<f64> {
    let mut result = DMatrix::<f64>::zeros(u.nrows(), vt.ncols());

    for k in 0..chi {
        let sigma = singular_values[k];

        for i in 0..u.nrows() {
            let value = u[(i, k)] * sigma;

            for j in 0..vt.ncols() {
                result[(i, j)] += value * vt[(k, j)];
            }
        }
    }

    result
}

// Theoretical relative error from discarded singular values.
fn tail_prediction(singular_values: &[f64], chi: usize) -> f64 {
    let total = singular_values.iter().map(|s| s * s).sum::<f64>();

    let discarded = singular_values.iter().skip(chi).map(|s| s * s).sum::<f64>();

    (discarded / total).sqrt()
}

// Generate the SVD figure.
fn make_plot(singular_values: &[f64], results: &[(usize, f64, f64)]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("artifacts/svd.png", (1400, 600)).into_drawing_area();

    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 2));

    // Plot 1: Singular spectrum.
    let max_s = singular_values[0] * 1.05;

    let mut chart1 = ChartBuilder::on(&areas[0])
        .caption("Singular Value Spectrum", ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..512f64, 0f64..max_s)?;

    chart1
        .configure_mesh()
        .x_desc("Singular-value index")
        .y_desc("Singular value")
        .draw()?;

    chart1.draw_series(LineSeries::new(
        (0..N).map(|i| ((i + 1) as f64, singular_values[i])),
        &BLUE,
    ))?;

    // Plot 2: Reconstruction error.
    let mut chart2 = ChartBuilder::on(&areas[1])
        .caption("SVD Reconstruction Error", ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..512f64, 0f64..1.0f64)?;

    chart2
        .configure_mesh()
        .x_desc("Retained rank chi")
        .y_desc("Relative Frobenius error")
        .draw()?;

    // Theoretical prediction for every rank.
    chart2
        .draw_series(LineSeries::new(
            (1..=N).map(|chi| (chi as f64, tail_prediction(singular_values, chi))),
            &BLUE,
        ))?
        .label("SVD tail prediction")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Measured reconstruction errors.
    chart2
        .draw_series(
            results
                .iter()
                .map(|(chi, measured, _)| Circle::new((*chi as f64, *measured), 5, RED.filled())),
        )?
        .label("Measured reconstruction")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.filled()));

    // 10% error threshold.
    chart2.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.1), (512.0, 0.1)],
        &RED,
    )))?;

    // Storage break-even point: chi = 256.
    chart2.draw_series(std::iter::once(PathElement::new(
        vec![(256.0, 0.0), (256.0, 1.0)],
        &BLACK,
    )))?;

    chart2
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("runs/svd")?;
    fs::create_dir_all("artifacts")?;

    println!("====================================");
    println!("Week 10 - Part 1: SVD Compression");
    println!("====================================");

    // 1. Generate Gaussian random matrix.
    println!("\n[1] Generating random matrix...");

    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    let mut original = DMatrix::<f64>::zeros(N, N);

    for i in 0..N {
        for j in 0..N {
            let value: f64 = StandardNormal.sample(&mut rng);
            original[(i, j)] = value;
        }
    }

    write_matrix("runs/svd/original_matrix.bin", &original)?;

    println!("Matrix generated: {} x {}", N, N);

    // 2. Perform SVD.
    println!("\n[2] Performing SVD...");

    let svd = original.clone().svd(true, true);

    let u = svd.u.expect("SVD failed to produce U");
    let vt = svd.v_t.expect("SVD failed to produce Vt");

    let singular_values: Vec<f64> = svd.singular_values.iter().copied().collect();

    if singular_values.windows(2).any(|w| w[0] < w[1]) {
        return Err("Singular values are not descending".into());
    }

    write_vector("runs/svd/singular_values.bin", &singular_values)?;

    println!("SVD completed.");

    // 3. Reload saved data for independent verification.
    println!("\n[3] Reloading saved evidence...");

    let saved_original = read_matrix("runs/svd/original_matrix.bin", N, N)?;

    let saved_singular_values = read_vector("runs/svd/singular_values.bin")?;

    // 4. Truncated SVD reconstruction.
    println!("\n[4] Truncated SVD experiments...");

    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    let mut csv = BufWriter::new(File::create("runs/svd/summary.csv")?);

    writeln!(
        csv,
        "chi,measured_error,predicted_error,absolute_difference,factor_entries,storage_ratio"
    )?;

    println!(
        "\n{:>6} {:>16} {:>16} {:>14} {:>14} {:>12}",
        "chi", "Measured", "Predicted", "Difference", "Entries", "Ratio"
    );

    let original_entries = N * N;

    for &chi in &RANKS {
        let reconstructed = reconstruct(&u, &singular_values, &vt, chi);

        let filename = format!("runs/svd/reconstruction_chi_{}.bin", chi,);

        write_matrix(&filename, &reconstructed)?;

        // Independently reload the reconstruction.
        let saved_reconstruction = read_matrix(&filename, N, N)?;

        // Actual error from saved matrices.
        let measured = relative_error(&saved_original, &saved_reconstruction);

        // Theoretical error from saved singular values.
        let predicted = tail_prediction(&saved_singular_values, chi);

        let difference = (measured - predicted).abs();

        let factor_entries = (N + N) * chi;

        let storage_ratio = factor_entries as f64 / original_entries as f64;

        println!(
            "{:>6} {:>16.10e} {:>16.10e} {:>14.4e} {:>14} {:>12.4}",
            chi, measured, predicted, difference, factor_entries, storage_ratio,
        );

        writeln!(
            csv,
            "{},{:.17e},{:.17e},{:.17e},{},{}",
            chi, measured, predicted, difference, factor_entries, storage_ratio,
        )?;

        if difference >= 1e-8 {
            return Err(format!("Error prediction mismatch at chi={}", chi).into());
        }

        if chi == N && measured >= 1e-8 {
            return Err("Full-rank reconstruction error exceeds 1e-8".into());
        }

        results.push((chi, measured, predicted));
    }

    csv.flush()?;

    // 5. Find first rank reaching 10% theoretical error.
    let crossing = (1..=N).find(|&chi| tail_prediction(&saved_singular_values, chi) <= 0.1);

    println!("\n[5] Storage analysis");

    println!("Original matrix entries: {}", original_entries,);

    println!("Storage break-even rank: 256",);

    if let Some(chi) = crossing {
        let entries = (N + N) * chi;

        println!("First rank reaching 10% error: {}", chi,);

        println!(
            "Storage ratio at this rank: {:.4}",
            entries as f64 / original_entries as f64,
        );
    }

    // 6. Save RNG metadata.
    fs::write(
        "runs/svd/metadata.txt",
        concat!(
            "Experiment: Week 10 Part 1 SVD\n",
            "Matrix: 512 x 512\n",
            "Distribution: Standard Normal\n",
            "RNG: ChaCha8Rng\n",
            "Seed: 2026 using seed_from_u64\n",
            "Sampling order: row-major, i then j\n",
            "rand: 0.8.5\n",
            "rand_chacha: 0.3.1\n",
            "rand_distr: 0.4.3\n",
            "Data format: raw little-endian f64\n",
            "Matrix layout: row-major\n",
        ),
    )?;

    // 7. Generate figure.
    println!("\n[6] Generating figure...");

    make_plot(&saved_singular_values, &results)?;

    println!("\n====================================");
    println!("Part 1 completed successfully!");
    println!("Figure: artifacts/svd.png");
    println!("Data: runs/svd/");
    println!("====================================");

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_svd_reconstruction() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

        let svd = matrix.clone().svd(true, true);

        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();

        let s: Vec<f64> = svd.singular_values.iter().copied().collect();

        let reconstructed = reconstruct(&u, &s, &vt, 3);

        assert!(relative_error(&matrix, &reconstructed) < 1e-10);
    }

    #[test]
    fn test_singular_values_order() {
        let matrix = DMatrix::<f64>::from_row_slice(2, 2, &[3.0, 1.0, 0.0, 2.0]);

        let svd = matrix.svd(true, true);

        let s: Vec<f64> = svd.singular_values.iter().copied().collect();

        assert!(s.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn test_tail_prediction() {
        let s = vec![4.0, 3.0, 2.0, 1.0];

        let p1 = tail_prediction(&s, 1);
        let p2 = tail_prediction(&s, 2);
        let p4 = tail_prediction(&s, 4);

        assert!(p2 < p1);
        assert!(p4.abs() < 1e-15);
    }

    #[test]
    fn test_matrix_file_roundtrip() {
        let matrix = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let path =
            std::env::temp_dir().join(format!("week10_roundtrip_{}.bin", std::process::id()));

        write_matrix(&path, &matrix).unwrap();

        let loaded = read_matrix(&path, 2, 3).unwrap();

        assert!(relative_error(&matrix, &loaded) < 1e-15);

        fs::remove_file(path).unwrap();
    }
}

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Embed the helper so the built executable is independent of its source path.
pub fn render_video(input: &Path, output: &Path, python: &Path, fps: u32) -> Result<(), String> {
    if fps == 0 || fps > 240 {
        return Err("fps must be between 1 and 240".into());
    }
    crate::read_run(input)?;
    let mut child = Command::new(python)
        .arg("-")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--fps")
        .arg(fps.to_string())
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start Python {}: {e}", python.display()))?;
    let write_result = child
        .stdin
        .take()
        .unwrap()
        .write_all(include_bytes!("../../scripts/animate.py"));
    let status = child.wait().map_err(|e| e.to_string())?;
    write_result.map_err(|e| format!("cannot send animation helper to Python: {e}"))?;
    if !status.success() {
        return Err(format!("video renderer failed ({status})"));
    }
    Ok(())
}

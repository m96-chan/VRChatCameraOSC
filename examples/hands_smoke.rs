//! Issue #8 phase-1 risk check: do the OpenCV Zoo MediaPipe-Hands ONNX
//! models (palm detection 192×192, hand landmarks 224×224) evaluate under
//! our candle-onnx fork, and how fast on CPU?
//!
//! ```bash
//! cargo run --release --example hands_smoke -- <palm.onnx> <landmarks.onnx>
//! ```
//! Feeds zero tensors — this checks op coverage and timing, not accuracy.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let palm = args
        .next()
        .context("usage: hands_smoke <palm.onnx> <landmarks.onnx>")?;
    let lms = args
        .next()
        .context("usage: hands_smoke <palm.onnx> <landmarks.onnx>")?;

    smoke("palm_detection", &palm, (1, 192, 192, 3))?;
    smoke("hand_landmarks", &lms, (1, 224, 224, 3))?;
    burn_bench("hand_landmarks", &lms, 224)?;
    // Optional third model: the pose estimation net (issue #28), 256×256.
    if let Some(pose) = args.next() {
        smoke("pose_estimation", &pose, (1, 256, 256, 3))?;
        burn_bench("pose_estimation", &pose, 256)?;
    }
    Ok(())
}

fn smoke(label: &str, path: &str, dims: (usize, usize, usize, usize)) -> Result<()> {
    let model: ModelProto = candle_onnx::read_file(path)?;
    let device = Device::Cpu;
    let consts = candle_onnx::initializer_tensors(&model)?
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_device(&device)?)))
        .collect::<candle_core::Result<HashMap<_, _>>>()?;
    let graph = model.graph.as_ref().context("no graph")?;
    let input = graph.input.first().context("no input")?;
    {
        let mut ops: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
        ops.sort();
        ops.dedup();
        eprintln!("{label}: op types: {ops:?}");
    }
    eprintln!(
        "{label}: input '{}', {} nodes, {} outputs: {:?}",
        input.name,
        graph.node.len(),
        graph.output.len(),
        graph
            .output
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>()
    );

    // Prefer the graph-declared input shape when fully static; otherwise
    // try NHWC (MediaPipe convention) then NCHW.
    let declared: Option<Vec<usize>> = input
        .r#type
        .as_ref()
        .and_then(|t| match &t.value {
            Some(candle_onnx::onnx::type_proto::Value::TensorType(tt)) => tt.shape.as_ref(),
            _ => None,
        })
        .map(|s| {
            s.dim
                .iter()
                .filter_map(|d| match &d.value {
                    Some(candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                        Some(*v as usize)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|v| v.len() == 4);
    let mut candidates: Vec<(&str, Vec<usize>)> = Vec::new();
    if let Some(d) = declared {
        candidates.push(("declared", d));
    }
    candidates.push(("NHWC", vec![dims.0, dims.1, dims.2, dims.3]));
    candidates.push(("NCHW", vec![dims.0, dims.3, dims.1, dims.2]));
    for (tag, shape) in candidates {
        let x = Tensor::zeros(shape.as_slice(), candle_core::DType::F32, &device)?;
        let mut inputs = consts.clone();
        inputs.insert(input.name.clone(), x);
        let t0 = Instant::now();
        match candle_onnx::simple_eval_with_initializers(&model, inputs) {
            Ok(outputs) => {
                let load_ms = t0.elapsed().as_millis();
                for (name, t) in outputs.iter() {
                    eprintln!("  [{tag}] out '{name}': {:?}", t.dims());
                }
                // Timed second run (first run may include one-time work).
                let mut inputs = consts.clone();
                inputs.insert(
                    input.name.clone(),
                    Tensor::zeros(shape.as_slice(), candle_core::DType::F32, &device)?,
                );
                let t1 = Instant::now();
                let _ = candle_onnx::simple_eval_with_initializers(&model, inputs)?;
                eprintln!(
                    "  [{tag}] OK — first {load_ms} ms, second {} ms",
                    t1.elapsed().as_millis()
                );
                return Ok(());
            }
            Err(e) => eprintln!("  [{tag}] failed: {e}"),
        }
    }
    anyhow::bail!("{label}: neither layout evaluated")
}

#[cfg(feature = "mesh-gpu")]
fn burn_bench(label: &str, path: &str, hw: usize) -> Result<()> {
    use vrchat_camera_osc::tracking::burn_onnx::BurnOnnx;
    let exec = BurnOnnx::from_path(path)?;
    let n = hw * hw * 3;
    let input: Vec<f32> = (0..n).map(|i| (i % 255) as f32 / 255.0).collect();
    // Warmup (shader compile / first upload), then timed runs.
    let _ = exec.run(input.clone(), [1, hw, hw, 3])?;
    let t0 = Instant::now();
    const RUNS: u32 = 20;
    for _ in 0..RUNS {
        let _ = exec.run(input.clone(), [1, hw, hw, 3])?;
    }
    eprintln!(
        "{label} on burn-wgpu: {:.1} ms/eval (avg of {RUNS})",
        t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64
    );
    Ok(())
}

#[cfg(not(feature = "mesh-gpu"))]
fn burn_bench(_label: &str, _path: &str, _hw: usize) -> Result<()> {
    eprintln!("mesh-gpu feature off — burn bench skipped");
    Ok(())
}

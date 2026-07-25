//! GPU FaceMesh stage on **burn-wgpu** (Vulkan/Metal/DX12, driver-only) —
//! ported from AvataCam `backends/mediapipe/burn_mesh.rs` (#62) for issue #17.
//!
//! A tiny runtime ONNX executor for the op subset FaceMesh V2 uses (`Conv`
//! incl. grouped + asymmetric pad, `PRelu`, `MaxPool`, `Add`, `Pad` incl.
//! channel-pad, `Sigmoid`, `Reshape`). The graph + weights are read from the
//! same ONNX file via candle-onnx's proto (no codegen); inference runs on
//! burn-wgpu. Weights are uploaded to the GPU **once at load**; each frame
//! only uploads the 256×256 crop and downloads 1434+1 floats.
//!
//! Why this exists: candle-onnx's per-op interpreter measured ~109 ms/frame
//! on CPU for this depthwise-separable-heavy graph and is launch/sync bound
//! on CUDA — nowhere near the 30 FPS realtime requirement. AvataCam hit the
//! same wall and validated this executor's output against the candle path.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use candle_onnx::onnx::{tensor_proto::DataType, ModelProto, NodeProto};

use burn::backend::Wgpu;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv2d, max_pool2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};

use super::mesh::{decode_landmarks, warp_crop, INPUT};
use super::roi::FaceRoi;
use super::{FaceMeshLandmarks, NUM_FACE_LANDMARKS};

type B = Wgpu;
type Dev = <B as Backend>::Device;

/// One parsed op. Weights are pre-uploaded to the GPU as burn tensors at load
/// time (cached) so each forward only clones cheap handles — no per-frame
/// upload.
///
/// Variant sizes differ wildly by design (Conv carries tensors, Add nothing);
/// a few hundred bytes x ~200 nodes is noise next to the weights themselves.
#[allow(clippy::large_enum_variant)]
enum Op {
    Conv {
        weight: Tensor<B, 4>,
        bias: Option<Tensor<B, 1>>,
        strides: [usize; 2],
        dil: [usize; 2],
        pads: [i64; 4],
        group: usize,
    },
    PRelu {
        slope: Tensor<B, 4>, // [1, C, 1, 1] for NCHW broadcast
    },
    Add,
    Sigmoid,
    MaxPool {
        k: [usize; 2],
        s: [usize; 2],
        pads: [i64; 4],
    },
    Pad {
        pads: [i64; 8],
        value: f32,
    },
    Reshape {
        shape: Vec<i64>,
    },
}

struct ExecNode {
    op: Op,
    act: Vec<String>, // activation inputs
    out: String,
}

pub struct BurnFaceMesh {
    device: Dev,
    nodes: Vec<ExecNode>,
    input_name: String,
    outputs: Vec<String>,
}

fn attr_ints(n: &NodeProto, name: &str) -> Vec<i64> {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
        .unwrap_or_default()
}
fn attr_int(n: &NodeProto, name: &str, default: i64) -> i64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

impl BurnFaceMesh {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let model: ModelProto = candle_onnx::read_file(path.as_ref())?;
        let graph = model.graph.as_ref().ok_or_else(|| anyhow!("no graph"))?;

        // Split initializers into f32 weights and i64 constants (pads/shapes).
        let mut weights: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut ints: HashMap<String, Vec<i64>> = HashMap::new();
        for t in &graph.initializer {
            let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
            match DataType::try_from(t.data_type) {
                Ok(DataType::Float) => {
                    let d = if !t.float_data.is_empty() {
                        t.float_data.clone()
                    } else {
                        t.raw_data
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect()
                    };
                    weights.insert(t.name.clone(), (d, shape));
                }
                Ok(DataType::Int64) => {
                    let d = if !t.int64_data.is_empty() {
                        t.int64_data.clone()
                    } else {
                        t.raw_data
                            .chunks_exact(8)
                            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                            .collect()
                    };
                    ints.insert(t.name.clone(), d);
                }
                _ => {}
            }
        }

        let input_name = graph
            .input
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow!("facemesh model has no input"))?;
        let outputs: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

        // Build (and GPU-upload) all weight tensors once, cached in the ops.
        let device = Dev::default();
        let t4 = |name: &str| -> Tensor<B, 4> {
            let (d, s) = &weights[name];
            Tensor::from_data(
                TensorData::new(d.clone(), [s[0], s[1], s[2], s[3]]),
                &device,
            )
        };
        let t1 = |name: &str| -> Tensor<B, 1> {
            let (d, s) = &weights[name];
            Tensor::from_data(
                TensorData::new(d.clone(), [s.iter().product::<usize>()]),
                &device,
            )
        };
        let slope = |name: &str| -> Tensor<B, 4> {
            let (d, s) = &weights[name];
            let c = s.iter().product::<usize>();
            Tensor::from_data(TensorData::new(d.clone(), [1, c, 1, 1]), &device)
        };

        let mut nodes = Vec::with_capacity(graph.node.len());
        for n in &graph.node {
            let f32_w: Vec<&String> = n
                .input
                .iter()
                .filter(|i| weights.contains_key(*i))
                .collect();
            let act: Vec<String> = n
                .input
                .iter()
                .filter(|i| !weights.contains_key(*i) && !ints.contains_key(*i))
                .cloned()
                .collect();
            let op = match n.op_type.as_str() {
                "Conv" => {
                    let s = attr_ints(n, "strides");
                    let d = attr_ints(n, "dilations");
                    let p = attr_ints(n, "pads");
                    Op::Conv {
                        weight: t4(f32_w[0]),
                        bias: f32_w.get(1).map(|b| t1(b)),
                        strides: [s[0] as usize, s[1] as usize],
                        dil: [d[0] as usize, d[1] as usize],
                        pads: pad4(&p),
                        group: attr_int(n, "group", 1) as usize,
                    }
                }
                "PRelu" => Op::PRelu {
                    slope: slope(f32_w[0]),
                },
                "Add" => Op::Add,
                "Sigmoid" => Op::Sigmoid,
                "MaxPool" => {
                    let k = attr_ints(n, "kernel_shape");
                    let s = attr_ints(n, "strides");
                    Op::MaxPool {
                        k: [k[0] as usize, k[1] as usize],
                        s: [s[0] as usize, s[1] as usize],
                        pads: pad4(&attr_ints(n, "pads")),
                    }
                }
                "Pad" => {
                    let pads = n
                        .input
                        .get(1)
                        .and_then(|i| ints.get(i))
                        .cloned()
                        .unwrap_or_else(|| attr_ints(n, "pads"));
                    let value = n
                        .input
                        .get(2)
                        .and_then(|i| weights.get(i))
                        .map(|(d, _)| d.first().copied().unwrap_or(0.0))
                        .unwrap_or(0.0);
                    let mut p8 = [0i64; 8];
                    for (i, v) in pads.iter().take(8).enumerate() {
                        p8[i] = *v;
                    }
                    Op::Pad { pads: p8, value }
                }
                "Reshape" => {
                    let shape = n
                        .input
                        .get(1)
                        .and_then(|i| ints.get(i))
                        .cloned()
                        .unwrap_or_default();
                    Op::Reshape { shape }
                }
                other => bail!("burn FaceMesh: unsupported op {other}"),
            };
            nodes.push(ExecNode {
                op,
                act,
                out: n.output[0].clone(),
            });
        }

        Ok(Self {
            device,
            nodes,
            input_name,
            outputs,
        })
    }

    pub fn run(
        &self,
        width: u32,
        height: u32,
        rgb: &[u8],
        roi: &FaceRoi,
    ) -> Result<FaceMeshLandmarks> {
        let buf = warp_crop(width, height, rgb, roi)?;
        let input =
            Tensor::<B, 4>::from_data(TensorData::new(buf, [1, 3, INPUT, INPUT]), &self.device);

        let mut env: HashMap<String, Tensor<B, 4>> = HashMap::new();
        env.insert(self.input_name.clone(), input);
        for node in &self.nodes {
            let out = self.eval(node, &env)?;
            env.insert(node.out.clone(), out);
        }

        // Landmarks = 1434-elem output; presence = a 1-elem output. Mirror
        // the candle CPU stage exactly: keep the most "present-looking"
        // (max) of the raw 1-element logits, then sigmoid; 1.0 if none.
        let mut raw: Option<Vec<f32>> = None;
        let mut presence_raw: Option<f32> = None;
        for name in &self.outputs {
            let Some(t) = env.get(name) else { continue };
            let n: usize = t.dims().iter().product();
            if n == NUM_FACE_LANDMARKS * 3 {
                raw = Some(
                    t.clone()
                        .into_data()
                        .to_vec()
                        .map_err(|e| anyhow!("{e:?}"))?,
                );
            } else if n == 1 {
                let v: Vec<f32> = t
                    .clone()
                    .into_data()
                    .to_vec()
                    .map_err(|e| anyhow!("{e:?}"))?;
                presence_raw = Some(presence_raw.map_or(v[0], |p: f32| p.max(v[0])));
            }
        }
        let raw = raw.ok_or_else(|| anyhow!("burn facemesh: 1434 output missing"))?;
        let presence = presence_raw
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .unwrap_or(1.0);
        Ok(decode_landmarks(&raw, presence, width, height, roi))
    }

    fn eval(&self, node: &ExecNode, env: &HashMap<String, Tensor<B, 4>>) -> Result<Tensor<B, 4>> {
        let act = |i: usize| env[&node.act[i]].clone();
        Ok(match &node.op {
            Op::Conv {
                weight,
                bias,
                strides,
                dil,
                pads,
                group,
            } => {
                let [hb, wb, he, we] = *pads;
                let (x, pad_hw) = if hb == he && wb == we {
                    (act(0), [hb as usize, wb as usize])
                } else {
                    (
                        act(0).pad((wb as usize, we as usize, hb as usize, he as usize), 0.0),
                        [0, 0],
                    )
                };
                conv2d(
                    x,
                    weight.clone(),
                    bias.clone(),
                    ConvOptions::new(*strides, pad_hw, *dil, *group),
                )
            }
            Op::PRelu { slope } => {
                let x = act(0);
                relu(x.clone()) - slope.clone() * relu(-x)
            }
            Op::Add => act(0) + act(1),
            Op::Sigmoid => sigmoid(act(0)),
            Op::MaxPool { k, s, pads } => {
                let pad = [pads[0] as usize, pads[1] as usize];
                max_pool2d(act(0), *k, *s, pad, [1, 1], false)
            }
            Op::Pad { pads, value } => {
                // [Nb,Cb,Hb,Wb, Ne,Ce,He,We]
                let (cb, ce) = (pads[1] as usize, pads[5] as usize);
                let (hb, he, wb, we) = (
                    pads[2] as usize,
                    pads[6] as usize,
                    pads[3] as usize,
                    pads[7] as usize,
                );
                let mut y = act(0);
                if hb + he + wb + we > 0 {
                    y = y.pad((wb, we, hb, he), *value);
                }
                if cb + ce > 0 {
                    let mut parts = Vec::new();
                    let d = y.dims();
                    if cb > 0 {
                        parts.push(Tensor::<B, 4>::zeros([d[0], cb, d[2], d[3]], &self.device));
                    }
                    parts.push(y);
                    if ce > 0 {
                        parts.push(Tensor::<B, 4>::zeros([d[0], ce, d[2], d[3]], &self.device));
                    }
                    y = Tensor::cat(parts, 1);
                }
                y
            }
            Op::Reshape { shape } => {
                let x = act(0);
                let mut shp = shape.clone();
                while shp.len() < 4 {
                    shp.insert(0, 1);
                }
                let total = x.dims().iter().product::<usize>() as i64;
                let known: i64 = shp.iter().filter(|&&v| v > 0).product();
                let dims4: [usize; 4] = std::array::from_fn(|i| {
                    if shp[i] < 0 {
                        (total / known.max(1)) as usize
                    } else {
                        shp[i] as usize
                    }
                });
                x.reshape(dims4)
            }
        })
    }
}

fn pad4(p: &[i64]) -> [i64; 4] {
    if p.len() == 4 {
        [p[0], p[1], p[2], p[3]]
    } else {
        [0; 4]
    }
}

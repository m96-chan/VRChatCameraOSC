//! Generic mini ONNX executor on **burn-wgpu** (Vulkan/Metal/DX12,
//! driver-only) — the runtime behind the GPU FaceMesh stage (issue #17,
//! ported from AvataCam #62) generalized for the hand landmark net
//! (issue #8): weights upload to the GPU once at load; each call uploads
//! one input tensor and downloads the outputs.
//!
//! Supported op subset (exactly what the MediaPipe-family graphs use):
//! `Conv` (grouped + asymmetric pad), `PRelu`, `MaxPool`, `Add`, `Pad`
//! (incl. channel-pad), `Sigmoid`, `Reshape`, and — for the hand landmark
//! net — `Transpose`, `Clip`, `GlobalAveragePool`, `Gemm`, `Squeeze`.
//! Every activation is carried as a rank-4 tensor (`[n,c,h,w]`-shaped, with
//! trailing 1s once the graph goes "flat"); `Gemm` reshapes internally.
//!
//! Why this exists: candle-onnx's per-op CPU interpreter measured ~109
//! ms/frame for FaceMesh and ~260 ms for the hand landmark net — nowhere
//! near realtime. Correctness is pinned by parity tests against the candle
//! path on identical inputs (`tests/mediapipe_onnx.rs`).

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

type B = Wgpu;
type Dev = <B as Backend>::Device;

/// One parsed op. Weights are pre-uploaded to the GPU as burn tensors at
/// load time (cached) so each forward only clones cheap handles.
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
    /// 4-D permutation (the TF→ONNX conversions' NHWC↔NCHW shims).
    Transpose {
        perm: [usize; 4],
    },
    /// Clamp to `[min, max]`.
    Clip {
        min: f32,
        max: f32,
    },
    /// Mean over the spatial dims, keepdims (EfficientNet-style head).
    GlobalAveragePool,
    /// `y = x·Wᵀ + b` on the flattened activation (rank juggling internal).
    Gemm {
        /// Stored pre-transposed to `[in, out]` so forward is a plain matmul.
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        out_features: usize,
    },
    /// Identity in the rank-4 world (trailing 1-dims carry the "squeezed"
    /// shape; output extraction reads flattened lengths).
    Squeeze,
}

struct ExecNode {
    op: Op,
    act: Vec<String>, // activation inputs
    out: String,
}

/// A loaded graph ready to run. See the module docs for scope.
pub struct BurnOnnx {
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
fn attr_f32(n: &NodeProto, name: &str) -> Option<f32> {
    n.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

fn pad4(p: &[i64]) -> [i64; 4] {
    if p.len() == 4 {
        [p[0], p[1], p[2], p[3]]
    } else {
        [0; 4]
    }
}

impl BurnOnnx {
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
            .ok_or_else(|| anyhow!("model has no input"))?;
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
                    let strides = attr_ints(n, "strides");
                    let dil = attr_ints(n, "dilations");
                    Op::Conv {
                        weight: t4(f32_w[0]),
                        bias: f32_w.get(1).map(|b| t1(b)),
                        strides: [
                            *strides.first().unwrap_or(&1) as usize,
                            *strides.get(1).unwrap_or(&1) as usize,
                        ],
                        dil: [
                            *dil.first().unwrap_or(&1) as usize,
                            *dil.get(1).unwrap_or(&1) as usize,
                        ],
                        pads: pad4(&attr_ints(n, "pads")),
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
                        s: [
                            *s.first().unwrap_or(&1) as usize,
                            *s.get(1).unwrap_or(&1) as usize,
                        ],
                        pads: pad4(&attr_ints(n, "pads")),
                    }
                }
                "Pad" => {
                    let pads_in = n
                        .input
                        .iter()
                        .find(|i| ints.contains_key(*i))
                        .and_then(|i| ints.get(i))
                        .cloned()
                        .unwrap_or_else(|| attr_ints(n, "pads"));
                    if pads_in.len() != 8 {
                        bail!("unsupported Pad rank: {pads_in:?}");
                    }
                    Op::Pad {
                        pads: pads_in.try_into().unwrap(),
                        value: 0.0,
                    }
                }
                "Reshape" => {
                    let shape = n
                        .input
                        .iter()
                        .find(|i| ints.contains_key(*i))
                        .and_then(|i| ints.get(i))
                        .cloned()
                        .unwrap_or_default();
                    Op::Reshape { shape }
                }
                "Transpose" => {
                    let perm = attr_ints(n, "perm");
                    if perm.len() != 4 {
                        bail!("unsupported Transpose perm: {perm:?}");
                    }
                    Op::Transpose {
                        perm: [
                            perm[0] as usize,
                            perm[1] as usize,
                            perm[2] as usize,
                            perm[3] as usize,
                        ],
                    }
                }
                "Clip" => {
                    // Opset <11 carries min/max as attributes; newer opsets
                    // pass them as (constant) inputs.
                    let mut lohi = f32_w.iter().map(|w| weights[*w].0[0]);
                    let min = attr_f32(n, "min").or_else(|| lohi.next());
                    let max = attr_f32(n, "max").or_else(|| lohi.next());
                    Op::Clip {
                        min: min.unwrap_or(f32::NEG_INFINITY),
                        max: max.unwrap_or(f32::INFINITY),
                    }
                }
                "GlobalAveragePool" => Op::GlobalAveragePool,
                "Gemm" => {
                    if attr_int(n, "transA", 0) != 0 {
                        bail!("Gemm transA unsupported");
                    }
                    let (wd, ws) = &weights[f32_w[0]];
                    if ws.len() != 2 {
                        bail!("Gemm weight must be 2-D, got {ws:?}");
                    }
                    // Store as [in, out] so forward is x[1,in]·W.
                    let trans_b = attr_int(n, "transB", 0) != 0;
                    let (rows, cols) = (ws[0], ws[1]);
                    let w2: Tensor<B, 2> =
                        Tensor::from_data(TensorData::new(wd.clone(), [rows, cols]), &device);
                    let (weight, out_features) = if trans_b {
                        // W is [out, in] -> transpose to [in, out].
                        (w2.transpose(), rows)
                    } else {
                        (w2, cols)
                    };
                    Op::Gemm {
                        weight,
                        bias: f32_w.get(1).map(|b| t1(b)),
                        out_features,
                    }
                }
                "Squeeze" => Op::Squeeze,
                other => bail!("burn executor: unsupported op {other}"),
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

    /// Run the graph on one rank-4 input; returns every graph output as
    /// `(name, flattened data)`.
    pub fn run(&self, input: Vec<f32>, dims: [usize; 4]) -> Result<Vec<(String, Vec<f32>)>> {
        let input = Tensor::<B, 4>::from_data(TensorData::new(input, dims), &self.device);
        let mut env: HashMap<String, Tensor<B, 4>> = HashMap::new();
        env.insert(self.input_name.clone(), input);
        for node in &self.nodes {
            let out = self.eval(node, &env)?;
            env.insert(node.out.clone(), out);
        }
        let mut result = Vec::with_capacity(self.outputs.len());
        for name in &self.outputs {
            let Some(t) = env.get(name) else { continue };
            let data: Vec<f32> = t
                .clone()
                .into_data()
                .to_vec()
                .map_err(|e| anyhow!("{e:?}"))?;
            result.push((name.clone(), data));
        }
        Ok(result)
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
            Op::Transpose { perm } => {
                let p: [isize; 4] = std::array::from_fn(|i| perm[i] as isize);
                act(0).permute(p)
            }
            Op::Clip { min, max } => act(0).clamp(*min, *max),
            Op::GlobalAveragePool => act(0).mean_dim(3).mean_dim(2),
            Op::Gemm {
                weight,
                bias,
                out_features,
            } => {
                let x = act(0);
                let n = x.dims().iter().product::<usize>();
                let x2: Tensor<B, 2> = x.reshape([1, n]);
                let mut y2 = x2.matmul(weight.clone());
                if let Some(b) = bias {
                    y2 = y2 + b.clone().reshape([1, *out_features]);
                }
                y2.reshape([1, *out_features, 1, 1])
            }
            Op::Squeeze => act(0),
        })
    }
}

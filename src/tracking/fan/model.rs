//! The FAN network itself (candle), a faithful port of `face_alignment/models/fan.py`.
//!
//! Module/field names mirror the PyTorch `state_dict` keys exactly so weights
//! exported from the reference load 1:1 via [`candle_nn::VarBuilder`]:
//! `conv1`, `bn1`, `conv2..conv4`, and per-stage `m{i}`, `top_m_{i}`,
//! `conv_last{i}`, `bn_end{i}`, `l{i}`, `bl{i}`, `al{i}`.
//!
//! Notes on parity with PyTorch:
//! - `conv3x3` uses `bias=False`; the 7x7 base conv and all 1x1 convs use bias.
//! - BatchNorm runs in eval mode (running stats), eps 1e-5.
//! - Downsampling is `F.avg_pool2d(x, 2, 2)`; upsampling is nearest, scale 2.

use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Module,
    ModuleT, VarBuilder,
};

const NUM_MODULES: usize = 4; // 2DFAN4
const HG_DEPTH: usize = 4;
const FEATURES: usize = 256;
const NUM_LANDMARKS: usize = 68;

fn conv3x3(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv2d> {
    // padding 1, stride 1, bias=False
    conv2d_no_bias(
        in_c,
        out_c,
        3,
        Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        },
        vb,
    )
}

fn conv1x1(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv2d> {
    // padding 0, stride 1, bias=True
    conv2d(in_c, out_c, 1, Conv2dConfig::default(), vb)
}

fn bn(features: usize, vb: VarBuilder) -> Result<BatchNorm> {
    batch_norm(features, BatchNormConfig::default(), vb)
}

/// Bottleneck residual block: BN-ReLU-Conv x3 with feature concatenation and an
/// optional 1x1 downsample on the residual path (used when in != out).
///
/// Public so the parity tests can build one directly from reference weights.
pub struct ConvBlock {
    bn1: BatchNorm,
    conv1: Conv2d,
    bn2: BatchNorm,
    conv2: Conv2d,
    bn3: BatchNorm,
    conv3: Conv2d,
    downsample: Option<(BatchNorm, Conv2d)>,
}

impl ConvBlock {
    pub fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let h2 = out_c / 2;
        let h4 = out_c / 4;
        let downsample = if in_c != out_c {
            let ds = vb.pp("downsample");
            // nn.Sequential(BatchNorm2d, ReLU, Conv2d) -> indices 0 and 2 carry params.
            Some((
                bn(in_c, ds.pp("0"))?,
                conv2d_no_bias(in_c, out_c, 1, Conv2dConfig::default(), ds.pp("2"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            bn1: bn(in_c, vb.pp("bn1"))?,
            conv1: conv3x3(in_c, h2, vb.pp("conv1"))?,
            bn2: bn(h2, vb.pp("bn2"))?,
            conv2: conv3x3(h2, h4, vb.pp("conv2"))?,
            bn3: bn(h4, vb.pp("bn3"))?,
            conv3: conv3x3(h4, h4, vb.pp("conv3"))?,
            downsample,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out1 = self.conv1.forward(&self.bn1.forward_t(x, false)?.relu()?)?;
        let out2 = self
            .conv2
            .forward(&self.bn2.forward_t(&out1, false)?.relu()?)?;
        let out3 = self
            .conv3
            .forward(&self.bn3.forward_t(&out2, false)?.relu()?)?;
        let cat = Tensor::cat(&[&out1, &out2, &out3], 1)?;
        let residual = match &self.downsample {
            Some((dbn, dconv)) => dconv.forward(&dbn.forward_t(x, false)?.relu()?)?,
            None => x.clone(),
        };
        cat + residual
    }
}

/// Recursive hourglass module (depth 4). Each level holds b1/b2/b3 conv blocks;
/// the innermost level additionally holds b2_plus.
pub struct HourGlass {
    // Indexed by level 1..=depth. Each entry: (b1, b2, b3).
    levels: Vec<(ConvBlock, ConvBlock, ConvBlock)>,
    b2_plus: ConvBlock, // at level 1
}

impl HourGlass {
    pub fn new(depth: usize, features: usize, vb: VarBuilder) -> Result<Self> {
        let mut levels = Vec::with_capacity(depth);
        for level in 1..=depth {
            let b1 = ConvBlock::new(features, features, vb.pp(format!("b1_{level}")))?;
            let b2 = ConvBlock::new(features, features, vb.pp(format!("b2_{level}")))?;
            let b3 = ConvBlock::new(features, features, vb.pp(format!("b3_{level}")))?;
            levels.push((b1, b2, b3));
        }
        let b2_plus = ConvBlock::new(features, features, vb.pp("b2_plus_1"))?;
        Ok(Self { levels, b2_plus })
    }

    fn forward_level(&self, level: usize, inp: &Tensor) -> Result<Tensor> {
        let (b1, b2, b3) = &self.levels[level - 1];
        let up1 = b1.forward(inp)?;

        let low1 = inp.avg_pool2d(2)?; // kernel 2, stride 2
        let low1 = b2.forward(&low1)?;

        let low2 = if level > 1 {
            self.forward_level(level - 1, &low1)?
        } else {
            self.b2_plus.forward(&low1)?
        };

        let low3 = b3.forward(&low2)?;

        let (h, w) = {
            let d = up1.dims4()?;
            (d.2, d.3)
        };
        let up2 = low3.upsample_nearest2d(h, w)?;
        up1 + up2
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_level(self.levels.len(), x)
    }
}

/// A single stacking stage of the FAN network.
struct Stage {
    m: HourGlass,
    top_m: ConvBlock,
    conv_last: Conv2d,
    bn_end: BatchNorm,
    l: Conv2d,
    // Present on all but the last stage.
    bl: Option<Conv2d>,
    al: Option<Conv2d>,
}

/// FAN (2DFAN4). Input `Nx3x256x256`, output the final stage's `Nx68x64x64`
/// heatmaps (all stages available via [`Fan::forward_all`]).
pub struct Fan {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: ConvBlock,
    conv3: ConvBlock,
    conv4: ConvBlock,
    stages: Vec<Stage>,
}

impl Fan {
    /// Build the network and load its weights from `vb`.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d(
            3,
            64,
            7,
            Conv2dConfig {
                padding: 3,
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let bn1 = bn(64, vb.pp("bn1"))?;
        let conv2 = ConvBlock::new(64, 128, vb.pp("conv2"))?;
        let conv3 = ConvBlock::new(128, 128, vb.pp("conv3"))?;
        let conv4 = ConvBlock::new(128, 256, vb.pp("conv4"))?;

        let mut stages = Vec::with_capacity(NUM_MODULES);
        for i in 0..NUM_MODULES {
            let m = HourGlass::new(HG_DEPTH, FEATURES, vb.pp(format!("m{i}")))?;
            let top_m = ConvBlock::new(256, 256, vb.pp(format!("top_m_{i}")))?;
            let conv_last = conv1x1(256, 256, vb.pp(format!("conv_last{i}")))?;
            let bn_end = bn(256, vb.pp(format!("bn_end{i}")))?;
            let l = conv1x1(256, NUM_LANDMARKS, vb.pp(format!("l{i}")))?;
            let (bl, al) = if i < NUM_MODULES - 1 {
                (
                    Some(conv1x1(256, 256, vb.pp(format!("bl{i}")))?),
                    Some(conv2d(
                        NUM_LANDMARKS,
                        256,
                        1,
                        Conv2dConfig::default(),
                        vb.pp(format!("al{i}")),
                    )?),
                )
            } else {
                (None, None)
            };
            stages.push(Stage {
                m,
                top_m,
                conv_last,
                bn_end,
                l,
                bl,
                al,
            });
        }
        Ok(Self {
            conv1,
            bn1,
            conv2,
            conv3,
            conv4,
            stages,
        })
    }

    /// Forward pass returning all `NUM_MODULES` heatmap stages (each `Nx68x64x64`).
    pub fn forward_all(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let x = self.bn1.forward_t(&self.conv1.forward(x)?, false)?.relu()?;
        let x = self.conv2.forward(&x)?.avg_pool2d(2)?;
        let x = self.conv3.forward(&x)?;
        let x = self.conv4.forward(&x)?;

        let mut previous = x;
        let mut outputs = Vec::with_capacity(self.stages.len());
        for (i, stage) in self.stages.iter().enumerate() {
            let hg = stage.m.forward(&previous)?;
            let ll = stage.top_m.forward(&hg)?;
            let ll = stage
                .bn_end
                .forward_t(&stage.conv_last.forward(&ll)?, false)?
                .relu()?;
            let tmp_out = stage.l.forward(&ll)?;

            if let (Some(bl), Some(al)) = (&stage.bl, &stage.al) {
                let ll_ = bl.forward(&ll)?;
                let tmp_out_ = al.forward(&tmp_out)?;
                previous = ((previous + ll_)? + tmp_out_)?;
            }
            outputs.push(tmp_out);
            let _ = i;
        }
        Ok(outputs)
    }

    /// Forward pass returning only the final heatmap stage (`Nx68x64x64`).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = self.forward_all(x)?;
        outputs
            .pop()
            .ok_or_else(|| candle_core::Error::Msg("FAN produced no output stage".into()))
    }
}

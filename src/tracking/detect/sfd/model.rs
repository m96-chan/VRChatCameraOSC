//! S3FD detector network (candle), a faithful port of
//! `face_alignment/detection/sfd/net_s3fd.py`.
//!
//! A VGG16-style conv stack (no BatchNorm) with three L2Norm-normalised
//! feature maps and six multi-scale detection heads. Field/module names mirror
//! the PyTorch `state_dict` so weights load 1:1: `conv1_1..conv7_2`, `fc6`,
//! `fc7`, `conv{3,4,5}_3_norm`, and the `*_mbox_conf` / `*_mbox_loc` heads.
//!
//! `forward` returns the 12 raw head tensors in the reference's order
//! `[cls1, reg1, cls2, reg2, ..., cls6, reg6]`, including the max-out background
//! trick on `cls1`. Softmax/decoding happens in [`super::decode`].

use candle_core::{Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

/// L2 normalisation across channels with a learned per-channel scale
/// (`net_s3fd.py::L2Norm`).
struct L2Norm {
    weight: Tensor, // [C]
    eps: f64,
}

impl L2Norm {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(channels, "weight")?;
        Ok(Self { weight, eps: 1e-10 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // norm = sqrt(sum(x^2, dim=1, keepdim)) + eps
        let norm = (x.sqr()?.sum_keepdim(1)?.sqrt()? + self.eps)?;
        let normed = x.broadcast_div(&norm)?;
        // multiply by weight reshaped to [1, C, 1, 1]
        let c = self.weight.dim(0)?;
        let w = self.weight.reshape((1, c, 1, 1))?;
        normed.broadcast_mul(&w)
    }
}

fn conv(
    in_c: usize,
    out_c: usize,
    k: usize,
    pad: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    conv2d(
        in_c,
        out_c,
        k,
        Conv2dConfig {
            padding: pad,
            stride,
            ..Default::default()
        },
        vb,
    )
}

/// The S3FD network.
pub struct S3fd {
    conv1_1: Conv2d,
    conv1_2: Conv2d,
    conv2_1: Conv2d,
    conv2_2: Conv2d,
    conv3_1: Conv2d,
    conv3_2: Conv2d,
    conv3_3: Conv2d,
    conv4_1: Conv2d,
    conv4_2: Conv2d,
    conv4_3: Conv2d,
    conv5_1: Conv2d,
    conv5_2: Conv2d,
    conv5_3: Conv2d,
    fc6: Conv2d,
    fc7: Conv2d,
    conv6_1: Conv2d,
    conv6_2: Conv2d,
    conv7_1: Conv2d,
    conv7_2: Conv2d,
    conv3_3_norm: L2Norm,
    conv4_3_norm: L2Norm,
    conv5_3_norm: L2Norm,
    conv3_3_norm_mbox_conf: Conv2d,
    conv3_3_norm_mbox_loc: Conv2d,
    conv4_3_norm_mbox_conf: Conv2d,
    conv4_3_norm_mbox_loc: Conv2d,
    conv5_3_norm_mbox_conf: Conv2d,
    conv5_3_norm_mbox_loc: Conv2d,
    fc7_mbox_conf: Conv2d,
    fc7_mbox_loc: Conv2d,
    conv6_2_mbox_conf: Conv2d,
    conv6_2_mbox_loc: Conv2d,
    conv7_2_mbox_conf: Conv2d,
    conv7_2_mbox_loc: Conv2d,
}

impl S3fd {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1_1: conv(3, 64, 3, 1, 1, vb.pp("conv1_1"))?,
            conv1_2: conv(64, 64, 3, 1, 1, vb.pp("conv1_2"))?,
            conv2_1: conv(64, 128, 3, 1, 1, vb.pp("conv2_1"))?,
            conv2_2: conv(128, 128, 3, 1, 1, vb.pp("conv2_2"))?,
            conv3_1: conv(128, 256, 3, 1, 1, vb.pp("conv3_1"))?,
            conv3_2: conv(256, 256, 3, 1, 1, vb.pp("conv3_2"))?,
            conv3_3: conv(256, 256, 3, 1, 1, vb.pp("conv3_3"))?,
            conv4_1: conv(256, 512, 3, 1, 1, vb.pp("conv4_1"))?,
            conv4_2: conv(512, 512, 3, 1, 1, vb.pp("conv4_2"))?,
            conv4_3: conv(512, 512, 3, 1, 1, vb.pp("conv4_3"))?,
            conv5_1: conv(512, 512, 3, 1, 1, vb.pp("conv5_1"))?,
            conv5_2: conv(512, 512, 3, 1, 1, vb.pp("conv5_2"))?,
            conv5_3: conv(512, 512, 3, 1, 1, vb.pp("conv5_3"))?,
            fc6: conv(512, 1024, 3, 3, 1, vb.pp("fc6"))?,
            fc7: conv(1024, 1024, 1, 0, 1, vb.pp("fc7"))?,
            conv6_1: conv(1024, 256, 1, 0, 1, vb.pp("conv6_1"))?,
            conv6_2: conv(256, 512, 3, 1, 2, vb.pp("conv6_2"))?,
            conv7_1: conv(512, 128, 1, 0, 1, vb.pp("conv7_1"))?,
            conv7_2: conv(128, 256, 3, 1, 2, vb.pp("conv7_2"))?,
            conv3_3_norm: L2Norm::load(256, vb.pp("conv3_3_norm"))?,
            conv4_3_norm: L2Norm::load(512, vb.pp("conv4_3_norm"))?,
            conv5_3_norm: L2Norm::load(512, vb.pp("conv5_3_norm"))?,
            conv3_3_norm_mbox_conf: conv(256, 4, 3, 1, 1, vb.pp("conv3_3_norm_mbox_conf"))?,
            conv3_3_norm_mbox_loc: conv(256, 4, 3, 1, 1, vb.pp("conv3_3_norm_mbox_loc"))?,
            conv4_3_norm_mbox_conf: conv(512, 2, 3, 1, 1, vb.pp("conv4_3_norm_mbox_conf"))?,
            conv4_3_norm_mbox_loc: conv(512, 4, 3, 1, 1, vb.pp("conv4_3_norm_mbox_loc"))?,
            conv5_3_norm_mbox_conf: conv(512, 2, 3, 1, 1, vb.pp("conv5_3_norm_mbox_conf"))?,
            conv5_3_norm_mbox_loc: conv(512, 4, 3, 1, 1, vb.pp("conv5_3_norm_mbox_loc"))?,
            fc7_mbox_conf: conv(1024, 2, 3, 1, 1, vb.pp("fc7_mbox_conf"))?,
            fc7_mbox_loc: conv(1024, 4, 3, 1, 1, vb.pp("fc7_mbox_loc"))?,
            conv6_2_mbox_conf: conv(512, 2, 3, 1, 1, vb.pp("conv6_2_mbox_conf"))?,
            conv6_2_mbox_loc: conv(512, 4, 3, 1, 1, vb.pp("conv6_2_mbox_loc"))?,
            conv7_2_mbox_conf: conv(256, 2, 3, 1, 1, vb.pp("conv7_2_mbox_conf"))?,
            conv7_2_mbox_loc: conv(256, 4, 3, 1, 1, vb.pp("conv7_2_mbox_loc"))?,
        })
    }

    /// Forward pass. Returns `[cls1, reg1, ..., cls6, reg6]` (raw, pre-softmax).
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mp = |t: &Tensor| t.max_pool2d(2); // kernel 2, stride 2

        let h = self.conv1_1.forward(x)?.relu()?;
        let h = self.conv1_2.forward(&h)?.relu()?;
        let h = mp(&h)?;

        let h = self.conv2_1.forward(&h)?.relu()?;
        let h = self.conv2_2.forward(&h)?.relu()?;
        let h = mp(&h)?;

        let h = self.conv3_1.forward(&h)?.relu()?;
        let h = self.conv3_2.forward(&h)?.relu()?;
        let h = self.conv3_3.forward(&h)?.relu()?;
        let f3_3 = h.clone();
        let h = mp(&h)?;

        let h = self.conv4_1.forward(&h)?.relu()?;
        let h = self.conv4_2.forward(&h)?.relu()?;
        let h = self.conv4_3.forward(&h)?.relu()?;
        let f4_3 = h.clone();
        let h = mp(&h)?;

        let h = self.conv5_1.forward(&h)?.relu()?;
        let h = self.conv5_2.forward(&h)?.relu()?;
        let h = self.conv5_3.forward(&h)?.relu()?;
        let f5_3 = h.clone();
        let h = mp(&h)?;

        let h = self.fc6.forward(&h)?.relu()?;
        let h = self.fc7.forward(&h)?.relu()?;
        let ffc7 = h.clone();
        let h = self.conv6_1.forward(&h)?.relu()?;
        let h = self.conv6_2.forward(&h)?.relu()?;
        let f6_2 = h.clone();
        let h = self.conv7_1.forward(&h)?.relu()?;
        let f7_2 = self.conv7_2.forward(&h)?.relu()?;

        let f3_3 = self.conv3_3_norm.forward(&f3_3)?;
        let f4_3 = self.conv4_3_norm.forward(&f4_3)?;
        let f5_3 = self.conv5_3_norm.forward(&f5_3)?;

        let cls1 = self.conv3_3_norm_mbox_conf.forward(&f3_3)?;
        let reg1 = self.conv3_3_norm_mbox_loc.forward(&f3_3)?;
        let cls2 = self.conv4_3_norm_mbox_conf.forward(&f4_3)?;
        let reg2 = self.conv4_3_norm_mbox_loc.forward(&f4_3)?;
        let cls3 = self.conv5_3_norm_mbox_conf.forward(&f5_3)?;
        let reg3 = self.conv5_3_norm_mbox_loc.forward(&f5_3)?;
        let cls4 = self.fc7_mbox_conf.forward(&ffc7)?;
        let reg4 = self.fc7_mbox_loc.forward(&ffc7)?;
        let cls5 = self.conv6_2_mbox_conf.forward(&f6_2)?;
        let reg5 = self.conv6_2_mbox_loc.forward(&f6_2)?;
        let cls6 = self.conv7_2_mbox_conf.forward(&f7_2)?;
        let reg6 = self.conv7_2_mbox_loc.forward(&f7_2)?;

        // max-out background label on cls1: bmax = max(c0, c1, c2); cls1 = [bmax, c3]
        let c0 = cls1.narrow(1, 0, 1)?;
        let c1 = cls1.narrow(1, 1, 1)?;
        let c2 = cls1.narrow(1, 2, 1)?;
        let c3 = cls1.narrow(1, 3, 1)?;
        let bmax = c0.maximum(&c1)?.maximum(&c2)?;
        let cls1 = Tensor::cat(&[&bmax, &c3], 1)?;

        Ok(vec![
            cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6,
        ])
    }
}

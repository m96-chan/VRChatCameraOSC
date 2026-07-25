//! Input preprocessing for FAN.
//!
//! - [`transform`] is a bit-exact port of `face_alignment/utils.py::transform`
//!   (affine map between image space and the 256-crop, plus its inverse). Used
//!   both to build the crop and to map heatmap coords back to the image.
//! - [`crop`] center-crops/scales an RGB image to the network input size,
//!   mirroring `utils.py::crop` (bilinear resize; not guaranteed bit-identical
//!   to OpenCV, so it is exercised end-to-end rather than for numeric parity).
//! - [`to_input_tensor`] packs a 256x256 RGB image into the `1x3x256x256`,
//!   `/255` float tensor the network expects (CHW, no mean/std).

use candle_core::{Device, Result, Tensor};
use image::RgbImage;

/// The face-crop scale factor (`200 * scale`), matching the reference.
pub const SCALE_FACTOR: f32 = 200.0;
/// FAN input resolution.
pub const INPUT_RES: u32 = 256;

/// Affine transform of a 2D point between image space and the crop, exactly as
/// `face_alignment/utils.py::transform`. Returns integer coordinates
/// (truncated toward zero, like numpy `astype(int)`).
///
/// The transform is a uniform scale + translation:
/// `a = resolution / (SCALE_FACTOR*scale)`, `t = resolution*(-center/h + 0.5)`.
pub fn transform(
    point: (f32, f32),
    center: (f32, f32),
    scale: f32,
    resolution: f32,
    invert: bool,
) -> (i64, i64) {
    let h = SCALE_FACTOR * scale;
    let a = resolution / h;
    let tx = resolution * (-center.0 / h + 0.5);
    let ty = resolution * (-center.1 / h + 0.5);

    let (nx, ny) = if invert {
        ((point.0 - tx) / a, (point.1 - ty) / a)
    } else {
        (a * point.0 + tx, a * point.1 + ty)
    };
    (nx as i64, ny as i64)
}

/// Center-crop and scale an RGB image to `resolution x resolution` around
/// `center` at the given `scale`. Port of `utils.py::crop`.
pub fn crop(image: &RgbImage, center: (f32, f32), scale: f32, resolution: u32) -> RgbImage {
    let res_f = resolution as f32;
    let ul = transform((1.0, 1.0), center, scale, res_f, true);
    let br = transform((res_f, res_f), center, scale, res_f, true);

    let new_w = br.0 - ul.0;
    let new_h = br.1 - ul.1;
    let (img_w, img_h) = (image.width() as i64, image.height() as i64);

    let mut canvas = RgbImage::new(new_w.max(1) as u32, new_h.max(1) as u32);

    // Overlap of the crop window with the source image (1-indexed math from the
    // reference, translated to 0-indexed slice bounds).
    let new_x = (1.max(-ul.0 + 1), br.0.min(img_w) - ul.0);
    let new_y = (1.max(-ul.1 + 1), br.1.min(img_h) - ul.1);
    let old_x = (1.max(ul.0 + 1), br.0.min(img_w));
    let old_y = (1.max(ul.1 + 1), br.1.min(img_h));

    for (ny, oy) in (new_y.0..=new_y.1).zip(old_y.0..=old_y.1) {
        let (nyi, oyi) = (ny - 1, oy - 1);
        if nyi < 0 || nyi >= new_h || oyi < 0 || oyi >= img_h {
            continue;
        }
        for (nx, ox) in (new_x.0..=new_x.1).zip(old_x.0..=old_x.1) {
            let (nxi, oxi) = (nx - 1, ox - 1);
            if nxi < 0 || nxi >= new_w || oxi < 0 || oxi >= img_w {
                continue;
            }
            let px = *image.get_pixel(oxi as u32, oyi as u32);
            canvas.put_pixel(nxi as u32, nyi as u32, px);
        }
    }

    image::imageops::resize(
        &canvas,
        resolution,
        resolution,
        image::imageops::FilterType::Triangle, // bilinear
    )
}

/// Pack a `resolution x resolution` RGB image into a `1x3xHxW` float tensor,
/// scaled to `[0,1]` (CHW order, no mean/std) — the FAN input convention.
pub fn to_input_tensor(image: &RgbImage, device: &Device) -> Result<Tensor> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for (x, y, px) in image.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        chw[y * w + x] = px[0] as f32 / 255.0;
        chw[h * w + y * w + x] = px[1] as f32 / 255.0;
        chw[2 * h * w + y * w + x] = px[2] as f32 / 255.0;
    }
    Tensor::from_vec(chw, (1, 3, h, w), device)
}

//! Checked surrogate backward for the selective SSM token mixer.
//!
//! This mirrors `src/nn/ssm.rs` one step at a time using f32 latent weights.
//! Multi-step BPTT composes this primitive by feeding `grad_prev_state` into
//! the previous timestep as its future state gradient.

use crate::learn::grad::{
    linear_forward_latent, linear_input_grad_latent, linear_input_grad_latent_add, GradBuffer,
    LatentWeights,
};

/// Dimensions for the one-step SSM surrogate.
#[derive(Debug, Clone, Copy)]
pub struct SsmDims {
    pub n_heads: usize,
    pub d_state: usize,
    pub d_head: usize,
}

impl SsmDims {
    pub fn d_model(self) -> usize {
        self.n_heads * self.d_head
    }

    pub fn state_len(self) -> usize {
        self.n_heads * self.d_state
    }

    pub fn x_proj_rows(self) -> usize {
        self.n_heads + 2 * self.state_len()
    }
}

/// Caller-owned cache for one SSM timestep.
#[derive(Debug, Clone)]
pub struct SsmStepCache {
    pub z: Vec<f32>,
    pub xbc: Vec<f32>,
    pub dt_raw: Vec<f32>,
    pub dt: Vec<f32>,
    pub x_avg: Vec<f32>,
    pub next_state: Vec<f32>,
    pub y: Vec<f32>,
}

impl SsmStepCache {
    pub fn new(dims: SsmDims) -> Self {
        Self {
            z: vec![0.0; dims.d_model()],
            xbc: vec![0.0; dims.x_proj_rows()],
            dt_raw: vec![0.0; dims.n_heads],
            dt: vec![0.0; dims.n_heads],
            x_avg: vec![0.0; dims.n_heads],
            next_state: vec![0.0; dims.state_len()],
            y: vec![0.0; dims.d_model()],
        }
    }
}

/// Additive gradient buffers for one SSM timestep.
pub struct SsmStepGrads {
    pub input: Vec<f32>,
    pub prev_state: Vec<f32>,
    pub in_proj: GradBuffer,
    pub x_proj: GradBuffer,
    pub out_proj: GradBuffer,
    pub a_log: Vec<f32>,
    pub d_param: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub z: Vec<f32>,
    pub xbc: Vec<f32>,
    pub y: Vec<f32>,
    pub state_total: Vec<f32>,
}

impl SsmStepGrads {
    pub fn new(dims: SsmDims) -> Self {
        let d_model = dims.d_model();
        Self {
            input: vec![0.0; d_model],
            prev_state: vec![0.0; dims.state_len()],
            in_proj: GradBuffer::new(d_model, d_model),
            x_proj: GradBuffer::new(dims.x_proj_rows(), d_model),
            out_proj: GradBuffer::new(d_model, d_model),
            a_log: vec![0.0; dims.state_len()],
            d_param: vec![0.0; dims.n_heads],
            dt_bias: vec![0.0; dims.n_heads],
            z: vec![0.0; d_model],
            xbc: vec![0.0; dims.x_proj_rows()],
            y: vec![0.0; d_model],
            state_total: vec![0.0; dims.state_len()],
        }
    }
}

fn softplus_like_ssm(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn softplus_grad_like_ssm(x: f32) -> f32 {
    if x > 20.0 {
        1.0
    } else if x < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

fn assert_shapes(
    dims: SsmDims,
    in_proj: &LatentWeights,
    x_proj: &LatentWeights,
    out_proj: &LatentWeights,
    a_log: &[f32],
    d_param: &[f32],
    dt_bias: &[f32],
) {
    let d_model = dims.d_model();
    assert_eq!(in_proj.rows, d_model);
    assert_eq!(in_proj.cols, d_model);
    assert_eq!(x_proj.rows, dims.x_proj_rows());
    assert_eq!(x_proj.cols, d_model);
    assert_eq!(out_proj.rows, d_model);
    assert_eq!(out_proj.cols, d_model);
    assert_eq!(a_log.len(), dims.state_len());
    assert_eq!(d_param.len(), dims.n_heads);
    assert_eq!(dt_bias.len(), dims.n_heads);
}

/// Dense f32 surrogate forward for one selective SSM timestep.
pub fn ssm_forward_latent(
    dims: SsmDims,
    in_proj: &LatentWeights,
    x_proj: &LatentWeights,
    out_proj: &LatentWeights,
    a_log: &[f32],
    d_param: &[f32],
    dt_bias: &[f32],
    input: &[f32],
    prev_state: &[f32],
    cache: &mut SsmStepCache,
    output: &mut [f32],
) {
    assert_shapes(dims, in_proj, x_proj, out_proj, a_log, d_param, dt_bias);
    let d_model = dims.d_model();
    assert_eq!(input.len(), d_model);
    assert_eq!(prev_state.len(), dims.state_len());
    assert_eq!(output.len(), d_model);

    linear_forward_latent(in_proj, input, &mut cache.z);
    linear_forward_latent(x_proj, &cache.z, &mut cache.xbc);
    cache.y.fill(0.0);

    let b_start = dims.n_heads;
    let c_start = b_start + dims.state_len();

    for h in 0..dims.n_heads {
        cache.dt_raw[h] = cache.xbc[h] + dt_bias[h];
        cache.dt[h] = softplus_like_ssm(cache.dt_raw[h]);

        let head_start = h * dims.d_head;
        let mut x_avg = 0.0f32;
        for d in 0..dims.d_head {
            x_avg += cache.z[head_start + d];
        }
        x_avg /= dims.d_head as f32;
        cache.x_avg[h] = x_avg;

        for s in 0..dims.d_state {
            let idx = h * dims.d_state + s;
            let a = a_log[idx].exp();
            let a_bar = (-a * cache.dt[h]).exp();
            let b_bar = cache.xbc[b_start + idx] * cache.dt[h];
            cache.next_state[idx] = a_bar * prev_state[idx] + b_bar * x_avg;
        }

        let mut y_h = 0.0f32;
        for s in 0..dims.d_state {
            let idx = h * dims.d_state + s;
            y_h += cache.xbc[c_start + idx] * cache.next_state[idx];
        }

        for d in 0..dims.d_head {
            cache.y[head_start + d] = y_h + d_param[h] * cache.z[head_start + d];
        }
    }

    linear_forward_latent(out_proj, &cache.y, output);
}

/// Backward for one selective SSM timestep.
///
/// `upstream_next_state` is the BPTT carry from the next timestep. Pass zeros
/// for a one-step loss that only depends on this timestep's output.
pub fn ssm_backward_latent(
    dims: SsmDims,
    in_proj: &LatentWeights,
    x_proj: &LatentWeights,
    out_proj: &LatentWeights,
    a_log: &[f32],
    d_param: &[f32],
    dt_bias: &[f32],
    input: &[f32],
    prev_state: &[f32],
    cache: &SsmStepCache,
    upstream_output: &[f32],
    upstream_next_state: &[f32],
    grads: &mut SsmStepGrads,
) {
    assert_shapes(dims, in_proj, x_proj, out_proj, a_log, d_param, dt_bias);
    let d_model = dims.d_model();
    assert_eq!(input.len(), d_model);
    assert_eq!(prev_state.len(), dims.state_len());
    assert_eq!(upstream_output.len(), d_model);
    assert_eq!(upstream_next_state.len(), dims.state_len());

    grads
        .out_proj
        .accumulate_ste(out_proj, &cache.y, upstream_output);
    linear_input_grad_latent(out_proj, upstream_output, &mut grads.y);

    let b_start = dims.n_heads;
    let c_start = b_start + dims.state_len();

    for h in 0..dims.n_heads {
        let head_start = h * dims.d_head;
        let mut grad_y_h = 0.0f32;
        for d in 0..dims.d_head {
            let gy = grads.y[head_start + d];
            grad_y_h += gy;
            grads.d_param[h] += gy * cache.z[head_start + d];
            grads.z[head_start + d] += gy * d_param[h];
        }

        for s in 0..dims.d_state {
            let idx = h * dims.d_state + s;
            let c_idx = c_start + idx;
            grads.xbc[c_idx] += grad_y_h * cache.next_state[idx];
            grads.state_total[idx] += upstream_next_state[idx] + grad_y_h * cache.xbc[c_idx];
        }

        let mut grad_dt = 0.0f32;
        let mut grad_x_avg = 0.0f32;
        for s in 0..dims.d_state {
            let idx = h * dims.d_state + s;
            let b_idx = b_start + idx;
            let state_grad = grads.state_total[idx];
            let a = a_log[idx].exp();
            let a_bar = (-a * cache.dt[h]).exp();
            let b_bar = cache.xbc[b_idx] * cache.dt[h];

            grads.prev_state[idx] += state_grad * a_bar;

            let grad_a_bar = state_grad * prev_state[idx];
            let grad_exp_arg = grad_a_bar * a_bar;
            grads.a_log[idx] += grad_exp_arg * (-cache.dt[h]) * a;
            grad_dt += grad_exp_arg * (-a);

            let grad_b_bar = state_grad * cache.x_avg[h];
            grads.xbc[b_idx] += grad_b_bar * cache.dt[h];
            grad_dt += grad_b_bar * cache.xbc[b_idx];
            grad_x_avg += state_grad * b_bar;
        }

        let grad_dt_raw = grad_dt * softplus_grad_like_ssm(cache.dt_raw[h]);
        grads.xbc[h] += grad_dt_raw;
        grads.dt_bias[h] += grad_dt_raw;

        let grad_z_avg = grad_x_avg / dims.d_head as f32;
        for d in 0..dims.d_head {
            grads.z[head_start + d] += grad_z_avg;
        }
    }

    grads.x_proj.accumulate_ste(x_proj, &cache.z, &grads.xbc);
    linear_input_grad_latent_add(x_proj, &grads.xbc, &mut grads.z);
    grads.in_proj.accumulate_ste(in_proj, input, &grads.z);
    linear_input_grad_latent_add(in_proj, &grads.z, &mut grads.input);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::check::{assert_gradient_close, GradCheckConfig};
    use crate::tensor::ternary::TernaryMatrix;

    fn latent_from_weights(rows: usize, cols: usize, weights: Vec<f32>) -> LatentWeights {
        assert_eq!(weights.len(), rows * cols);
        let trits = vec![0i8; rows * cols];
        let scales = vec![1.0f32; rows];
        LatentWeights {
            weights,
            ternary: TernaryMatrix::pack(&trits, &scales, rows, cols),
            scales,
            rows,
            cols,
            version: 0,
        }
    }

    fn dims() -> SsmDims {
        SsmDims {
            n_heads: 2,
            d_state: 2,
            d_head: 2,
        }
    }

    fn ssm_params(dims: SsmDims) -> Vec<f32> {
        let d_model = dims.d_model();
        let state_len = dims.state_len();
        let in_len = d_model * d_model;
        let x_len = dims.x_proj_rows() * d_model;
        let out_len = d_model * d_model;
        let n = in_len + x_len + out_len + state_len + dims.n_heads * 2;
        let mut params = Vec::with_capacity(n);

        for i in 0..in_len + x_len + out_len {
            params.push(((i as i32 % 9) - 4) as f32 * 0.06);
        }
        for i in 0..state_len {
            params.push(-1.2 + i as f32 * 0.13);
        }
        for i in 0..dims.n_heads {
            params.push(0.7 + i as f32 * 0.2);
        }
        for i in 0..dims.n_heads {
            params.push(-0.2 + i as f32 * 0.15);
        }

        params
    }

    fn split_params(
        p: &[f32],
        dims: SsmDims,
    ) -> (
        LatentWeights,
        LatentWeights,
        LatentWeights,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let d_model = dims.d_model();
        let state_len = dims.state_len();
        let in_len = d_model * d_model;
        let x_len = dims.x_proj_rows() * d_model;
        let out_len = d_model * d_model;
        assert_eq!(
            p.len(),
            in_len + x_len + out_len + state_len + dims.n_heads * 2
        );

        let in_proj = latent_from_weights(d_model, d_model, p[0..in_len].to_vec());
        let x_start = in_len;
        let x_proj = latent_from_weights(
            dims.x_proj_rows(),
            d_model,
            p[x_start..x_start + x_len].to_vec(),
        );
        let out_start = x_start + x_len;
        let out_proj =
            latent_from_weights(d_model, d_model, p[out_start..out_start + out_len].to_vec());
        let a_start = out_start + out_len;
        let a_log = p[a_start..a_start + state_len].to_vec();
        let d_start = a_start + state_len;
        let d_param = p[d_start..d_start + dims.n_heads].to_vec();
        let dt_start = d_start + dims.n_heads;
        let dt_bias = p[dt_start..].to_vec();

        (in_proj, x_proj, out_proj, a_log, d_param, dt_bias)
    }

    fn forward_output(
        params: &[f32],
        dims: SsmDims,
        input: &[f32],
        prev_state: &[f32],
    ) -> (SsmStepCache, Vec<f32>) {
        let (in_proj, x_proj, out_proj, a_log, d_param, dt_bias) = split_params(params, dims);
        let mut cache = SsmStepCache::new(dims);
        let mut output = vec![0.0f32; dims.d_model()];
        ssm_forward_latent(
            dims,
            &in_proj,
            &x_proj,
            &out_proj,
            &a_log,
            &d_param,
            &dt_bias,
            input,
            prev_state,
            &mut cache,
            &mut output,
        );
        (cache, output)
    }

    #[test]
    fn ssm_input_and_state_grads_match_finite_difference() {
        let dims = dims();
        let params = ssm_params(dims);
        let (in_proj, x_proj, out_proj, a_log, d_param, dt_bias) = split_params(&params, dims);
        let input = vec![0.6f32, -0.3, 0.8, -0.5];
        let prev_state = vec![0.2f32, -0.1, 0.3, -0.4];
        let target = vec![10.0f32, -7.5, 5.0, -2.5];
        let future_state_grad = vec![1.0f32, -0.75, 0.5, -0.25];
        let mut input_and_state = input.clone();
        input_and_state.extend_from_slice(&prev_state);

        assert_gradient_close(
            &input_and_state,
            |p| {
                let x = &p[0..dims.d_model()];
                let prev = &p[dims.d_model()..];
                let (cache, output) = forward_output(&params, dims, x, prev);
                let output_loss: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum();
                let future_loss: f64 = cache
                    .next_state
                    .iter()
                    .zip(future_state_grad.iter())
                    .map(|(&s, &g)| s as f64 * g as f64)
                    .sum();
                output_loss + future_loss
            },
            |p, out| {
                let x = &p[0..dims.d_model()];
                let prev = &p[dims.d_model()..];
                let mut cache = SsmStepCache::new(dims);
                let mut output = vec![0.0f32; dims.d_model()];
                ssm_forward_latent(
                    dims,
                    &in_proj,
                    &x_proj,
                    &out_proj,
                    &a_log,
                    &d_param,
                    &dt_bias,
                    x,
                    prev,
                    &mut cache,
                    &mut output,
                );
                let upstream_output: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grads = SsmStepGrads::new(dims);
                ssm_backward_latent(
                    dims,
                    &in_proj,
                    &x_proj,
                    &out_proj,
                    &a_log,
                    &d_param,
                    &dt_bias,
                    x,
                    prev,
                    &cache,
                    &upstream_output,
                    &future_state_grad,
                    &mut grads,
                );
                for i in 0..dims.d_model() {
                    out[i] = grads.input[i] as f64;
                }
                for i in 0..dims.state_len() {
                    out[dims.d_model() + i] = grads.prev_state[i] as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 6e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn ssm_params_match_finite_difference() {
        let dims = dims();
        let params = ssm_params(dims);
        let input = vec![0.6f32, -0.3, 0.8, -0.5];
        let prev_state = vec![0.2f32, -0.1, 0.3, -0.4];
        let target = vec![10.0f32, -7.5, 5.0, -2.5];
        let future_state_grad = vec![1.0f32, -0.75, 0.5, -0.25];

        assert_gradient_close(
            &params,
            |p| {
                let (cache, output) = forward_output(p, dims, &input, &prev_state);
                let output_loss: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum();
                let future_loss: f64 = cache
                    .next_state
                    .iter()
                    .zip(future_state_grad.iter())
                    .map(|(&s, &g)| s as f64 * g as f64)
                    .sum();
                output_loss + future_loss
            },
            |p, out| {
                let (in_proj, x_proj, out_proj, a_log, d_param, dt_bias) = split_params(p, dims);
                let mut cache = SsmStepCache::new(dims);
                let mut output = vec![0.0f32; dims.d_model()];
                ssm_forward_latent(
                    dims,
                    &in_proj,
                    &x_proj,
                    &out_proj,
                    &a_log,
                    &d_param,
                    &dt_bias,
                    &input,
                    &prev_state,
                    &mut cache,
                    &mut output,
                );
                let upstream_output: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grads = SsmStepGrads::new(dims);
                ssm_backward_latent(
                    dims,
                    &in_proj,
                    &x_proj,
                    &out_proj,
                    &a_log,
                    &d_param,
                    &dt_bias,
                    &input,
                    &prev_state,
                    &cache,
                    &upstream_output,
                    &future_state_grad,
                    &mut grads,
                );

                let mut cursor = 0usize;
                for &g in grads.in_proj.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grads.x_proj.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grads.out_proj.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grads.a_log.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grads.d_param.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grads.dt_bias.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                assert_eq!(cursor, out.len());
            },
            GradCheckConfig {
                epsilon: 1e-2,
                tolerance: 2e-2,
                denom_floor: 1e-8,
            },
        );
    }
}

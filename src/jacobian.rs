// Copyright 2018 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::pert::*;
use crate::EPS_F64;

pub fn forward_jacobian_vec_f64(x: &Vec<f64>, fs: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (fs)(&x);
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (fs)(&x1);
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / EPS_F64.sqrt())
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[cfg(feature = "ndarray")]
pub fn forward_jacobian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (fs)(&x);
    let rn = fx.len();
    let n = x.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = x.clone();
        x1[i] += EPS_F64.sqrt();
        let fx1 = (fs)(&x1);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
        }
    }
    out
}

pub fn central_jacobian_vec_f64(x: &Vec<f64>, fs: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            let mut x2 = x.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (fs)(&x1);
            let fx2 = (fs)(&x2);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[cfg(feature = "ndarray")]
pub fn central_jacobian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    // TODO: get rid of this!
    let fx = (fs)(&x);
    let rn = fx.len();
    let n = x.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = x.clone();
        let mut x2 = x.clone();
        x1[i] += EPS_F64.sqrt();
        x2[i] -= EPS_F64.sqrt();
        let fx1 = (fs)(&x1);
        let fx2 = (fs)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    out
}

pub fn forward_jacobian_vec_prod_vec_f64(
    x: &Vec<f64>,
    fs: &Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let fx = (fs)(&x);
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + EPS_F64.sqrt() * pi)
        .collect();
    let fx1 = (fs)(&x1);
    fx1.iter()
        .zip(fx.iter())
        .map(|(a, b)| (a - b) / EPS_F64.sqrt())
        .collect::<Vec<f64>>()
}

#[cfg(feature = "ndarray")]
pub fn forward_jacobian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let fx = (fs)(&x);
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + EPS_F64.sqrt() * pi)
        .collect();
    let fx1 = (fs)(&x1);
    (fx1 - fx) / EPS_F64.sqrt()
}

pub fn central_jacobian_vec_prod_vec_f64(
    x: &Vec<f64>,
    fs: &Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + EPS_F64.sqrt() * pi)
        .collect();
    let x2 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi - EPS_F64.sqrt() * pi)
        .collect();
    let fx1 = (fs)(&x1);
    let fx2 = (fs)(&x2);
    fx1.iter()
        .zip(fx2.iter())
        .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
        .collect::<Vec<f64>>()
}

#[cfg(feature = "ndarray")]
pub fn central_jacobian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + EPS_F64.sqrt() * pi)
        .collect();
    let x2 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi - EPS_F64.sqrt() * pi)
        .collect();
    let fx1 = (fs)(&x1);
    let fx2 = (fs)(&x2);
    (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
}

pub fn forward_jacobian_pert_vec_f64(
    x: &Vec<f64>,
    fs: &Fn(&Vec<f64>) -> Vec<f64>,
    pert: PerturbationVectors,
) -> Vec<Vec<f64>> {
    let fx = (fs)(&x);
    let mut out = vec![vec![0.0; x.len()]; fx.len()];
    for pert_item in pert.iter() {
        let mut x1 = x.clone();
        for j in pert_item.x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
        }
        let fx1 = (fs)(&x1);
        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[*x_idx][*j] = (fx1[*j] - fx[*j]) / EPS_F64.sqrt();
            }
        }
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_jacobian_pert_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: PerturbationVectors,
) -> ndarray::Array2<f64> {
    let fx = (fs)(&x);
    let mut out = ndarray::Array2::zeros((fx.len(), x.len()));
    for pert_item in pert.iter() {
        let mut x1 = x.clone();
        for j in pert_item.x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
        }
        let fx1 = (fs)(&x1);
        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[(*x_idx, *j)] = (fx1[*j] - fx[*j]) / EPS_F64.sqrt();
            }
        }
    }
    out
}

pub fn central_jacobian_pert_vec_f64(
    x: &Vec<f64>,
    fs: &Fn(&Vec<f64>) -> Vec<f64>,
    pert: PerturbationVectors,
) -> Vec<Vec<f64>> {
    let mut out = vec![];
    for (i, pert_item) in pert.iter().enumerate() {
        let mut x1 = x.clone();
        let mut x2 = x.clone();
        for j in pert_item.x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
            x2[*j] -= EPS_F64.sqrt();
        }
        let fx1 = (fs)(&x1);
        let fx2 = (fs)(&x2);
        if i == 0 {
            out = vec![vec![0.0; x.len()]; fx1.len()];
        }
        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[*x_idx][*j] = (fx1[*j] - fx2[*j]) / (2.0 * EPS_F64.sqrt());
            }
        }
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn central_jacobian_pert_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: PerturbationVectors,
) -> ndarray::Array2<f64> {
    let mut out = ndarray::Array2::zeros((1, 1));
    for (i, pert_item) in pert.iter().enumerate() {
        let mut x1 = x.clone();
        let mut x2 = x.clone();
        for j in pert_item.x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
            x2[*j] -= EPS_F64.sqrt();
        }
        let fx1 = (fs)(&x1);
        let fx2 = (fs)(&x2);
        if i == 0 {
            out = ndarray::Array2::zeros((fx1.len(), x.len()));
        }
        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[(*x_idx, *j)] = (fx1[*j] - fx2[*j]) / (2.0 * EPS_F64.sqrt());
            }
        }
    }
    out
}

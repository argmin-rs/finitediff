// Copyright 2018 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::EPS_F64;

pub fn forward_hessian_vec_f64(x: &Vec<f64>, grad: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (grad)(x);
    let n = x.len();
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (grad)(&x1);
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[i][j] + out[j][i]) / 2.0;
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (grad)(&x);
    let rn = fx.len();
    let n = x.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = x.clone();
        x1[i] += EPS_F64.sqrt();
        let fx1 = (grad)(&x1);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
        }
    }
    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[(i, j)] + out[(j, i)]) / 2.0;
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    out
}

pub fn central_hessian_vec_f64(x: &Vec<f64>, grad: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            let mut x2 = x.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (grad)(&x1);
            let fx2 = (grad)(&x2);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[i][j] + out[j][i]) / 2.0;
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn central_hessian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    // TODO: get rid of this!
    let fx = (grad)(&x);
    let rn = fx.len();
    let n = x.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = x.clone();
        let mut x2 = x.clone();
        x1[i] += EPS_F64.sqrt();
        x2[i] -= EPS_F64.sqrt();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[(i, j)] + out[(j, i)]) / 2.0;
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    out
}

pub fn forward_hessian_vec_prod_vec_f64(
    x: &Vec<f64>,
    grad: &Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let fx = (grad)(x);
    let out: Vec<f64> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        fx1.iter()
            .zip(fx.iter())
            .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let fx = (grad)(&x);
    let rn = fx.len();
    let mut out = ndarray::Array1::zeros(rn);
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
        .collect();
    let fx1 = (grad)(&x1);
    for j in 0..rn {
        out[j] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
    }
    out
}

pub fn central_hessian_vec_prod_vec_f64(
    x: &Vec<f64>,
    grad: &Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let out: Vec<f64> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
            .collect();
        let x2 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi - pi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        fx1.iter()
            .zip(fx2.iter())
            .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

#[cfg(feature = "ndarray")]
pub fn central_hessian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let rn = x.len();
    let mut out = ndarray::Array1::zeros(rn);
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
        .collect();
    let x2 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi - pi * EPS_F64.sqrt())
        .collect();
    let fx1 = (grad)(&x1);
    let fx2 = (grad)(&x2);
    for j in 0..rn {
        out[j] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
    }
    out
}

pub fn forward_hessian_nograd_vec_f64(x: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<Vec<f64>> {
    // TODO: f(x + EPS * e_i) needs to be precomputed!!
    let fx = (f)(x);
    let n = x.len();
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let t = {
                let mut xi = x.clone();
                xi[i] += EPS_F64.sqrt();
                let mut xj = x.clone();
                xj[j] += EPS_F64.sqrt();
                let mut xij = x.clone();
                xij[i] += EPS_F64.sqrt();
                xij[j] += EPS_F64.sqrt();
                let fxi = (f)(&xi);
                let fxj = (f)(&xj);
                let fxij = (f)(&xij);
                (fxij - fxi - fxj + fx) / EPS_F64
            };
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_nograd_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array2<f64> {
    // TODO: f(x + EPS * e_i) needs to be precomputed!!
    let fx = (f)(x);
    let n = x.len();
    let mut out = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let t = {
                let mut xi = x.clone();
                xi[i] += EPS_F64.sqrt();
                let mut xj = x.clone();
                xj[j] += EPS_F64.sqrt();
                let mut xij = x.clone();
                xij[i] += EPS_F64.sqrt();
                xij[j] += EPS_F64.sqrt();
                let fxi = (f)(&xi);
                let fxj = (f)(&xj);
                let fxij = (f)(&xij);
                (fxij - fxi - fxj + fx) / EPS_F64
            };
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    out
}

pub fn forward_hessian_nograd_sparse_vec_f64(
    x: &Vec<f64>,
    f: &Fn(&Vec<f64>) -> f64,
    indices: Vec<(usize, usize)>,
) -> Vec<Vec<f64>> {
    let fx = (f)(x);
    let n = x.len();
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for (i, j) in indices {
        let t = {
            let mut xi = x.clone();
            xi[i] += EPS_F64.sqrt();
            let mut xj = x.clone();
            xj[j] += EPS_F64.sqrt();
            let mut xij = x.clone();
            xij[i] += EPS_F64.sqrt();
            xij[j] += EPS_F64.sqrt();
            let fxi = (f)(&xi);
            let fxj = (f)(&xj);
            let fxij = (f)(&xij);
            (fxij - fxi - fxj + fx) / EPS_F64
        };
        out[i][j] = t;
        out[j][i] = t;
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_nograd_sparse_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
    indices: Vec<(usize, usize)>,
) -> ndarray::Array2<f64> {
    let fx = (f)(x);
    let n = x.len();
    let mut out = ndarray::Array2::zeros((n, n));
    for (i, j) in indices {
        let t = {
            let mut xi = x.clone();
            xi[i] += EPS_F64.sqrt();
            let mut xj = x.clone();
            xj[j] += EPS_F64.sqrt();
            let mut xij = x.clone();
            xij[i] += EPS_F64.sqrt();
            xij[j] += EPS_F64.sqrt();
            let fxi = (f)(&xi);
            let fxj = (f)(&xj);
            let fxij = (f)(&xij);
            (fxij - fxi - fxj + fx) / EPS_F64
        };
        out[(i, j)] = t;
        out[(j, i)] = t;
    }
    out
}

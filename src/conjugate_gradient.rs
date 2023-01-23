use crate::linear_algebra as la;
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Conjugate Gradient method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x_0` - Start point.
/// * `b` - Vector b.
/// * `eps` - error threshold.
pub fn cg(a: ArrayView2<f64>, x_0: ArrayView1<f64>, b: ArrayView1<f64>, eps: f64) -> Array1<f64> {
    let n = x_0.len();
    let mut x = x_0.to_owned();
    let mut r = Array1::zeros(n);
    let mut p = Array1::zeros(n);

    // initialize: r_0, p_0
    for i in 0..n {
        let mut ri = b[i];
        for j in 0..n {
            ri -= a[[i, j]] * x_0[j];
        }
        r[i] = ri;
    }

    for i in 0..n {
        p[i] = r[i];
    }

    loop {
        let ap = a.dot(&p);
        let rr_ = r.dot(&r);
        let alpha = rr_ / p.dot(&ap);

        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
        }
        println!("x(k+1)={:?}", x);

        for i in 0..n {
            r[i] = r[i] - alpha * ap[i];
        }

        if la::norm_l2(&r) < eps {
            return x;
        }

        let beta = r.dot(&r) / rr_;

        // update p to the next conjugate vector
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::is_convergent_l1norm;
    use ndarray::{arr1, arr2};

    #[test]
    fn cg_1x1() {
        let eps = 1e-9;
        let a = arr2(&[[1.0]]);
        let b = arr1(&[2.0]);
        let x_0 = arr1(&[0.0]);

        let x_k = cg(a.view(), x_0.view(), b.view(), eps);
        let truth = arr1(&[2.0]);
        assert!(is_convergent_l1norm(x_k.view(), truth.view(), eps));
    }

    #[test]
    fn cg_2x2() {
        let eps = 1e-9;
        let a = arr2(&[[4.0, -1.0], [-1.0, 1.0]]);
        let b = arr1(&[4.0, 2.0]);
        let x_0 = arr1(&[0.0, 0.0]);

        let x_k = cg(a.view(), x_0.view(), b.view(), eps);
        let truth = arr1(&[2.0, 4.0]);
        assert!(is_convergent_l1norm(x_k.view(), truth.view(), eps));
    }
}

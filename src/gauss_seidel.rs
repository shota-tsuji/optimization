use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::assert::{is_convergent_l1norm};

/// Gauss-Seidel method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x` - Vector x.
/// * `b` - Vector b.
pub fn gauss_seidel(
    a: ArrayView2<f64>,
    x_0: ArrayView1<f64>,
    b: ArrayView1<f64>,
    eps: f64,
) -> Array1<f64> {
    let mut x_k1 = x_0.to_owned();
    let mut x_k_ = x_k1.clone();
    let n = x_0.len();
    loop {
        for i in 0..n {
            let mut xi = b[i];
            for j in 0..n {
                if j != i {
                    xi -= a[[i, j]] * x_k1[j];
                }
            }
            x_k1[i] = xi / a[[i, i]];
        }

        if is_convergent_l1norm(x_k1.view(), x_k_.view(), eps) {
            return x_k1;
        }

        //println!("{:?}", x_k1);
        for i in 0..n {
            x_k_[i] = x_k1[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::is_convergent_l1norm;
    use ndarray::{arr1, arr2};

    #[test]
    fn gauss_seidel_2x2() {
        let eps = 1e-9;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = arr1(&[13.0, 8.0]);
        let x_0 = arr1(&[0.0, 0.0]);

        let x_k = gauss_seidel(a.view(), x_0.view(), b.view(), eps);
        let truth = arr1(&[1.0, 2.0]);
        assert!(is_convergent_l1norm(x_k.view(), truth.view(), eps));
    }

    #[test]
    fn gauss_seidel_3x3() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = arr1(&[0.0, 4.0, 6.0]);
        let x_0 = arr1(&[0.0, 0.0, 0.0]);

        let x_k = gauss_seidel(a.view(), x_0.view(), b.view(), eps);
        let truth = arr1(&[-1.0, 1.0, 2.0]);
        assert!(is_convergent_l1norm(x_k.view(), truth.view(), eps));
    }
}

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::assert::is_convergent_l1norm;

/// Jacobi method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x` - Vector x.
/// * `b` - Vector b.
pub fn jacobi(
    a: ArrayView2<f64>,
    x_0: ArrayView1<f64>,
    b: ArrayView1<f64>,
    eps: f64,
) -> Array1<f64> {
    let mut x_k1 = x_0.to_owned();
    let mut y = x_k1.clone();
    let n = x_0.len();
    loop {
        for i in 0..n {
            let mut xi = b[i];
            for j in 0..n {
                if j != i {
                    xi -= a[[i, j]] * y[j];
                }
            }
            x_k1[i] = xi / a[[i, i]];
        }

        //println!("x(k+1)={:?}, x(k)={:?}", &x_k1, &y);
        if is_convergent_l1norm(x_k1.view(), y.view(), eps) {
            return x_k1;
        }

        for i in 0..n {
            y[i] = x_k1[i];
        }
    }
}

pub fn is_diagonally_dominant(a: ArrayView2<f64>) -> bool {
    for i in 0..a.shape()[0] {
        let a_ii = a[[i, i]];
        let mut non_diagonals = 0.0;
        for j in 0..a.shape()[1] {
            if j != i {
                non_diagonals += a[[i, j]].abs();
            }
        }

        if a_ii.abs() <= non_diagonals {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::is_convergent_l1norm;
    use ndarray::{arr1, arr2};

    #[test]
    fn jacobi_1x1() {
        let eps = 1e-10;
        let a = arr2(&[[1.0]]);
        let b = arr1(&[2.0]);
        let x_0 = arr1(&[0.0]);
        let x_k1 = jacobi(a.view(), x_0.view(), b.view(), eps);
        let ans = arr1(&vec![2.0]);
        assert!(is_convergent_l1norm(ans.view(), x_k1.view(), eps));
    }

    #[test]
    fn jacobi_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = arr1(&[13.0, 8.0]);
        let x_0 = arr1(&[0.0, 0.0]);
        let x_k1 = jacobi(a.view(), x_0.view(), b.view(), eps);
        let ans = arr1(&vec![1.0, 2.0]);
        assert!(is_convergent_l1norm(ans.view(), x_k1.view(), eps));
    }

    #[test]
    fn jacobi_3x3_eps6() {
        let eps = 1e-6;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = arr1(&[0.0, 4.0, 6.0]);
        let x_0 = arr1(&[0.0, 0.0, 0.0]);
        let x_k1 = jacobi(a.view(), x_0.view(), b.view(), eps);
        let ans = arr1(&vec![-1.0, 1.0, 2.0]);
        assert!(is_convergent_l1norm(ans.view(), x_k1.view(), eps));
    }

    #[test]
    fn jacobi_3x3_eps10() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = arr1(&[0.0, 4.0, 6.0]);
        let x_0 = arr1(&[0.0, 0.0, 0.0]);
        let x_k1 = jacobi(a.view(), x_0.view(), b.view(), eps);
        let ans = arr1(&vec![-1.0, 1.0, 2.0]);
        assert!(is_convergent_l1norm(ans.view(), x_k1.view(), eps));
    }

    #[test]
    fn diagonally_dominant_2x2() {
        let a = arr2(&[[3.0, -2.0], [1.0, -3.0]]);
        assert!(is_diagonally_dominant(a.view()));
    }

    #[test]
    fn not_diagonally_dominant_2x2() {
        let a = arr2(&[[0.0, 1.0], [0.0, 1.0]]);
        assert!(!is_diagonally_dominant(a.view()));
    }

    #[test]
    fn diagonally_dominant_3x3() {
        let a = arr2(&[[3.0, -1.0, 1.0], [1.0, -3.0, 1.0], [-1.0, 2.0, 4.0]]);
        assert!(is_diagonally_dominant(a.view()));
    }
}

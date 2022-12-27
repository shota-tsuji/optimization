use ndarray::ArrayView2;

use crate::assert::is_convergent;

/// Gauss-Seidel method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x` - Vector x.
/// * `b` - Vector b.
pub fn gauss_seidel(a: ArrayView2<f64>, x_0: &Vec<f64>, b: &Vec<f64>, eps: f64) -> Vec<f64> {
    let mut x_k1 = x_0.clone();
    let mut x_k_ = x_k1.clone();
    loop {
        for i in 0..x_k1.len() {
            let mut xi = b[i];
            for j in 0..x_k1.len() {
                if j != i {
                    xi -= a[[i, j]] * x_k1[j];
                }
            }
            x_k1[i] = xi / a[[i, i]];
        }

        if is_convergent(&x_k1, &x_k_, eps) {
            return x_k1;
        }

        println!("{:?}", x_k1);
        for i in 0..x_k1.len() {
            x_k_[i] = x_k1[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn gauss_seidel_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = vec![13.0, 8.0];
        let x_0 = vec![0.0, 0.0];
        let x_k = &gauss_seidel(a.view(), &x_0, &b, eps);
        assert!(
            is_convergent(&vec![1.0, 2.0], x_k, eps),
            "eps={}, x={:?}",
            eps,
            x_k
        );
    }

    #[test]
    fn gauss_seidel_3x3() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = vec![0.0, 4.0, 6.0];
        let x_0 = vec![0.0, 0.0, 0.0];
        let x_k = &gauss_seidel(a.view(), &x_0, &b, eps);
        assert!(
            is_convergent(&vec![-1.0, 1.0, 2.0], x_k, eps),
            "eps={}, x={:?}",
            eps,
            x_k
        );
    }
}

use ndarray::{Array1, Array2};

/// Calculate x y^T
///
/// Return N x N matrix.
/// * `x` - N-elements vector.
/// * `y` - N-elements vector.
pub fn outer(x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
    let m = x.len();
    let n = y.len();
    let mut matrix = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            matrix[[i, j]] = x[i] * y[j];
        }
    }

    matrix
}

pub fn copy(x: &mut Array1<f64>, y: &Array1<f64>) {
    let n = x.len();
    for i in 0..n {
        x[i] = y[i];
    }
}

pub fn norm_l2(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

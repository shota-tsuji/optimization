use ndarray::ArrayView1;

/// check whether convergent
///
/// returns true if satisfy convergence, otherwise returns false.
///
/// * `x_k1` - Vector x^(k+1).
/// * `x_k_` - Vector x^(k).
/// * `eps` - epsilon (error).
pub fn is_convergent(x_k1: &Vec<f64>, x_k_: &Vec<f64>, eps: f64) -> bool {
    let mut delta_sum = 0.0;
    for i in 0..x_k1.len() {
        delta_sum += (x_k1[i] - x_k_[i]).abs();
    }

    if delta_sum < eps {
        true
    } else {
        false
    }
}

pub fn norm_l1(v: ArrayView1<f64>) -> f64 {
    let mut delta_sum = 0.0;
    for i in 0..v.len() {
        delta_sum += v[i].abs();
    }

    delta_sum
}

/// check whether convergent
///
/// returns true if satisfy convergence, otherwise returns false.
///
/// * `x_k1` - Vector x^(k+1).
/// * `x_k_` - Vector x^(k).
/// * `eps` - epsilon (error).
pub fn is_convergent_l1norm(x_k1: ArrayView1<f64>, x_k_: ArrayView1<f64>, eps: f64) -> bool {
    let v = &x_k1.to_owned() - &x_k_.to_owned();
    if norm_l1(v.view()) < eps {
        true
    } else {
        false
    }
}

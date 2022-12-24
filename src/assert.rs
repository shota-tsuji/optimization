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
        delta_sum += ((x_k1[i] - x_k_[i]) / x_k1[i]).abs();
    }

    if delta_sum < eps {
        true
    } else {
        false
    }
}

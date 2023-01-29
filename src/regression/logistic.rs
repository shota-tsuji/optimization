use argmin::core::{CostFunction, Error, Gradient};
use ndarray::{Array1, Array2};

pub struct Logistic {
    y: Array1<i8>,
    features: Array2<f64>,
    // number of feature parameters
    pub n: usize,
    // number of train data
    pub l: usize,
}

impl Logistic {
    pub fn new(y: Array1<i8>, features: Array2<f64>) -> Logistic {
        let l = y.len();
        let n = features.shape()[1];
        Logistic { y, features, n, l }
    }
}

impl CostFunction for Logistic {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut sum = 0.0;

        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += p[j] * self.features[[i, j]];
            }
            let yi = f64::from(self.y[i]);
            sum += f64::ln(1.0 + f64::exp(wx)) - yi * wx;
        }

        Ok(sum)
    }
}

impl Gradient for Logistic {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut g = Array1::zeros(self.n);
        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += p[j] * self.features[[i, j]];
            }

            let yi = f64::from(self.y[i]);

            for j in 0..self.n {
                let p = 1.0 / (1.0 + f64::exp(-wx));
                g[j] += self.features[[i, j]] * (p - yi);
            }
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array};

    #[test]
    fn new() {
        // if w is 0 vector, sum of loss becomes l * log(2).
        let l = 10;
        let n = 1;
        let y = Array::zeros(l);
        let w = Array1::<f64>::zeros(n);
        let mat_x = Array::zeros((l, n));
        let logistic = Logistic::new(y, mat_x);

        assert_eq!(6.931471805599453, logistic.cost(&w).unwrap());
    }

    #[test]
    fn gradient() {
        // if w is 0 vector
        let n = 2;
        let y = arr1(&[-1, -1]);
        let w = Array1::<f64>::zeros(n);
        let mat_x = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        let logistic = Logistic::new(y, mat_x);

        assert_eq!(arr1(&[3.0, 0.0]), logistic.gradient(&w).unwrap());
    }
}

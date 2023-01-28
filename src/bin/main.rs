use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin::solver::quasinewton::BFGS;

use ndarray::{Array1, Array2};

use optimization::utils;

fn main() {
    // divide to labels and samples
    let path = String::from("./dummy.csv");
    let (y, mat_x) = utils::load_csv(path);
    // unique
    // index array for value == -1
    let y_nega_idx: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, &val)| val == -1)
        .map(|(i, _)| i)
        .collect();
    // ones-array
    // zero with index-array
    let mut y_bin = Array1::ones(y.len());
    for i in y_nega_idx {
        y_bin[i] = 0;
    }

    let logistic = Logistic::new(y_bin, mat_x.clone());
    let mut regression = Regression {};
    regression.train(logistic);
}

struct Logistic {
    y: Array1<i8>,
    features: Array2<f64>,
    // number of feature parameters
    n: usize,
    // number of train data
    l: usize,
}

impl Logistic {
    fn new(y: Array1<i8>, features: Array2<f64>) -> Logistic {
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

struct Regression {}

impl Regression {
    fn train(&mut self, logistic: Logistic) -> Result<(), Error> {
        let init_param = Array1::<f64>::zeros(logistic.n);
        let init_hessian = Array2::<f64>::eye(logistic.n);

        let linesearch = HagerZhangLineSearch::new();
        let solver = BFGS::new(linesearch);

        let res = Executor::new(logistic, solver)
            .configure(|state| {
                state
                    .param(init_param)
                    .inv_hessian(init_hessian)
                    .max_iters(60)
            })
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        std::thread::sleep(std::time::Duration::from_secs(1));

        println!("{res}");
        Ok(())
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

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin::solver::quasinewton::BFGS;

use ndarray::{Array, Array1, Array2};

fn main() {
    // ラベルと特徴量に分ける
    let path = String::from("./dummy.csv");
    let (y, mat_x) = load_csv(path);
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

    let mut logistic = Logistic::new(y_bin, mat_x.clone());
    let mut regression = Regression {};
    regression.train(logistic);
}

fn load_csv(path: String) -> (Array1<i8>, Array2<f64>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .unwrap();
    let n = rdr.records().count();
    //println!("{}", n);
    let mut y = Array::zeros(n);
    let mut features = Array::zeros((n, 123));

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .unwrap();
    for (i, record) in rdr.records().enumerate() {
        let record = record.unwrap();
        y[i] = record[0].parse::<i8>().unwrap();
        for (j, data) in record.iter().enumerate() {
            if j > 0 {
                let index = data.parse::<usize>().unwrap() - 1;
                features[[i, index]] = 1.0;
            }
        }
    }

    (y, features)
}

struct Logistic {
    y: Array1<i8>,
    features: Array2<f64>,
    // number of feature parameters
    n: usize,
    // number of train data
    l: usize,
    num_newton_step: i64,
    num_quasi_newton: i64,
}

impl Logistic {
    fn new(y: Array1<i8>, features: Array2<f64>) -> Logistic {
        let l = y.len();
        let n = features.shape()[1];
        let num_newton_step = 0;
        let num_quasi_newton = 0;
        Logistic {
            y,
            features,
            n,
            l,
            num_newton_step,
            num_quasi_newton,
        }
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
        //for i in 0..g.len() {
        //    g[i] = 0.0;
        //}
        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                // todo: modify features as parameter.
                wx += p[j] * self.features[[i, j]];
            }

            let yi = f64::from(self.y[i]);

            for j in 0..self.n {
                let p = 1.0 / (1.0 + f64::exp(-wx));
                //g[j] += -yi * self.features[[i, j]] / (1.0 + f64::exp(yi * wx));
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
    use ndarray::{arr1, arr2};

    #[test]
    fn new() {
        // if w is 0 vector, sum of loss becomes l * log(2).
        let l = 10;
        let n = 1;
        let y = Array::zeros(l);
        let features = Array::zeros((l, n));
        let regression = Logistic::new(y, features.clone());
        let w = Array1::<f64>::zeros(n);
        assert_eq!(6.931471805599453, regression.cost(&w).unwrap());
    }

    #[test]
    fn derivative() {
        // if w is 0 vector
        let n = 2;
        let y = arr1(&[-1, -1]);
        let features = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        let regression = Logistic::new(y, features);
        let w = Array1::<f64>::zeros(n);

        assert_eq!(arr1(&[3.0, 0.0]), regression.gradient(&w).unwrap());
    }

    #[test]
    fn wolfe_conditions() {}
}

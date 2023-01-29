pub mod logistic;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{Error, Executor, IterState};
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin::solver::quasinewton::BFGS;
use ndarray::{Array1, Array2};

use logistic::Logistic;

pub struct Regression {}

impl Regression {
    pub fn train(
        &mut self,
        logistic: Logistic,
    ) -> Result<IterState<Array1<f64>, Array1<f64>, (), Array2<f64>, f64>, Error> {
        let w0 = Array1::<f64>::zeros(logistic.n);
        let h_inv = Array2::<f64>::eye(logistic.n);

        let solver = BFGS::new(HagerZhangLineSearch::new());

        let res = Executor::new(logistic, solver)
            .configure(|state| state.param(w0).inv_hessian(h_inv).max_iters(60))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        std::thread::sleep(std::time::Duration::from_secs(1));

        println!("{res}");
        Ok(res.state().clone())
    }
}

use rand::Rng;
use rstat::univariate::cauchy::Params;
use rv::prelude::{ContinuousDistr, Sampleable};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag="type")]
pub enum Distribution {
    Cauchy{loc: f64, scale: f64},
    LogNormal{s: f64, loc: f64, scale: f64},
    Zero
}

impl Distribution{
    pub fn probability(&self, x: f64) -> f64 {
        match self {
            Distribution::Cauchy { loc, scale } => {
                rv::dist::Cauchy::new(*loc, *scale).unwrap().pdf(&x)
            }
            Distribution::LogNormal { s, loc, scale } => {
                rv::dist::LogNormal::new(scale.ln(), *s).unwrap().pdf(&(x - loc))
            },
            Distribution::Zero => 0.
        }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            Distribution::Cauchy { loc, scale } => {
                rv::dist::Cauchy::new(*loc, *scale).unwrap().draw(rng)
            }
            Distribution::LogNormal { s, loc, scale } => {
                // https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma
                let x: f64 = rv::dist::LogNormal::new(scale.ln(), *s).unwrap().draw(rng);
                x + loc
            },
            Distribution::Zero => panic!("Attempted sampling from Zero distribution.")
        }
    }
}

//! Pure f64 inference. This module has no solver dependency or LP calls.
use super::*;
use serde::Deserialize;

pub(super) const FEATURE_NAMES: [&str;12] = ["x_val","frac_dist","field_h","abs_field_h",
    "deg_norm","bond_bias","eff_field","abs_eff_field","mean_bond_dual","mean_bond_gap",
    "mean_nbr_frac","gap_ratio"];

#[derive(Clone,Deserialize)]
struct Stats { mean:Vec<f64>, std:Vec<f64> }
#[derive(Clone,Deserialize)]
struct Weights { w1:Vec<Vec<f64>>, b1:Vec<f64>, w2:Vec<f64>, b2:f64 }
#[derive(Clone,Deserialize)]
struct Probe { features:Vec<Vec<f64>>, logits:Vec<f64>, probabilities:Vec<f64> }
#[derive(Clone,Deserialize)]
pub(super) struct Network {
    schema:String, feature_names:Vec<String>, stats:Stats, weights:Weights, probes:Vec<Probe>,
}
#[derive(Serialize)]
pub(super) struct Parity {
    pub probe_count:usize, pub max_probability_error:f64, pub max_logit_error:f64,
}

impl Network {
    pub fn load(path:&PathBuf) -> Result<Self> {
        let model:Self=serde_json::from_slice(&fs::read(path)?)?;
        model.validate()?;
        Ok(model)
    }
    fn validate(&self) -> Result<()> {
        check(self.schema=="branchopt.mlp.v1","unsupported model schema")?;
        check(self.feature_names.iter().map(String::as_str).eq(FEATURE_NAMES.iter().copied()),
              "model feature order does not match Rust")?;
        check(self.stats.mean.len()==12 && self.stats.std.len()==12,"invalid feature statistics")?;
        let w=&self.weights;
        check(w.w1.len()==12 && w.w1.iter().all(|row|row.len()==48)
              && w.b1.len()==48 && w.w2.len()==48,"network must have architecture 12-48-1")?;
        check(self.stats.mean.iter().all(|v|v.is_finite())
              && self.stats.std.iter().all(|v|v.is_finite() && *v>=0.0)
              && w.w1.iter().flatten().chain(w.b1.iter()).chain(w.w2.iter()).all(|v|v.is_finite())
              && w.b2.is_finite(),"non-finite model coefficient")
    }
    pub fn scores(&self,features:&[Vec<f64>]) -> Result<Vec<f64>> {
        let mut result=Vec::with_capacity(features.len());
        for row in features {
            check(row.len()==12 && row.iter().all(|v|v.is_finite()),"invalid neural feature row")?;
            let mut input=[0.0;12];
            for f in 0..12 {input[f]=(row[f]-self.stats.mean[f])/self.stats.std[f].max(1e-6);}
            let mut score=self.weights.b2;
            for h in 0..48 {
                let mut value=self.weights.b1[h];
                for f in 0..12 {value+=input[f]*self.weights.w1[f][h];}
                score+=self.weights.w2[h]*value.tanh();
            }
            check(score.is_finite(),"non-finite neural score")?;
            result.push(score);
        }
        Ok(result)
    }
    pub fn probabilities(&self,features:&[Vec<f64>]) -> Result<Vec<f64>> {
        let scores=self.scores(features)?;
        check(!scores.is_empty(),"empty candidate softmax")?;
        let top=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);
        let mut p:Vec<f64>=scores.iter().map(|s|(s-top).exp()).collect();
        let sum=p.iter().sum::<f64>();
        for v in &mut p {*v/=sum;}
        Ok(p)
    }
    pub fn verify_probes(&self) -> Result<Parity> {
        check(self.probes.len()==6,"model must contain exactly six validation probes")?;
        let (mut prob_error,mut logit_error)=(0.0_f64,0.0_f64);
        for probe in &self.probes {
            let logits=self.scores(&probe.features)?;
            let probs=self.probabilities(&probe.features)?;
            check(probe.features.len()>=2 && probe.logits.len()==logits.len()
                  && probe.probabilities.len()==probs.len(),"invalid probe dimensions")?;
            check(probe.logits.iter().chain(probe.probabilities.iter()).all(|v|v.is_finite()),
                  "non-finite JAX probe")?;
            for (a,b) in logits.iter().zip(&probe.logits) {logit_error=logit_error.max((a-b).abs());}
            for (a,b) in probs.iter().zip(&probe.probabilities) {prob_error=prob_error.max((a-b).abs());}
        }
        Ok(Parity{probe_count:self.probes.len(),max_probability_error:prob_error,max_logit_error:logit_error})
    }
}

#[cfg(test)]
pub(super) fn adversarial_fixture() -> Network {
    let mut w1=vec![vec![0.0;48];12];
    w1[2][0]=100.0; w1[4][1]=-50.0;
    let mut w2=vec![0.0;48]; w2[0]=1.0; w2[1]=1.0;
    Network{schema:"branchopt.mlp.v1".into(),feature_names:FEATURE_NAMES.iter().map(|s|s.to_string()).collect(),
        stats:Stats{mean:vec![0.0;12],std:vec![1.0;12]},
        weights:Weights{w1,b1:vec![0.0;48],w2,b2:0.0},probes:Vec::new()}
}

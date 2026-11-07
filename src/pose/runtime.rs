//! The inference runtime seam.
//!
//! This is the only file that talks to the ONNX runtime (`tract`). The rest
//! of the pose code sees a [`Network`] (a loaded, optimized model shared
//! between threads) and [`Session`]s (one per thread, holding the mutable run
//! state), with plain `f32` tensors in and out, and an [`InferencePool`]
//! (the threads one inference may use). Another runtime slots in behind the
//! same types.

use std::sync::Arc;

use tract_linalg::multithread::{Executor, multithread_tract_scope};
use tract_onnx::prelude::{
    DatumExt, Framework, InferenceModelExt, IntoTValue, Tensor, TypedSimplePlan, TypedSimpleState,
    tvec,
};

use crate::{Error, Result};

/// A model output: its shape and values in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Output {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

/// A loaded model with a fixed `[1, 3, height, width]` float input.
#[derive(Debug, Clone)]
pub(crate) struct Network {
    name: String,
    input: [usize; 4],
    plan: Arc<TypedSimplePlan>,
}

impl Network {
    /// Parses and optimizes an ONNX model for a fixed input shape.
    ///
    /// Shape annotations inside the graph (`value_info`, output shapes) are
    /// ignored: the mmdeploy exports carry a symbolic batch size there that
    /// conflicts with the fixed input; tract infers every shape from the
    /// input instead.
    pub(crate) fn load(name: &str, onnx: &[u8], input: [usize; 4]) -> Result<Self> {
        let err = |e: tract_onnx::prelude::TractError| Error::model(name, format!("{e:#}"));
        let mut reader = onnx;
        let plan = tract_onnx::onnx()
            .with_ignore_value_info(true)
            .with_ignore_output_shapes(true)
            .model_for_read(&mut reader)
            .and_then(|m| m.with_input_fact(0, f32::fact(input).into()))
            .and_then(InferenceModelExt::into_optimized)
            .and_then(tract_onnx::prelude::IntoRunnable::into_runnable)
            .map_err(err)?;
        Ok(Self {
            name: name.to_owned(),
            input,
            plan,
        })
    }

    /// A new run state. Sessions share the network's weights.
    pub(crate) fn session(&self) -> Result<Session> {
        let state = self
            .plan
            .spawn()
            .map_err(|e| Error::model(&self.name, format!("{e:#}")))?;
        Ok(Session {
            network: self.clone(),
            state,
            pool: InferencePool::calling_thread(),
        })
    }
}

/// The cores one inference may use: a thread pool, shared (cheaply cloned)
/// between the models that run one after another on the same thread, e.g.
/// a tracker's detector and pose model.
#[derive(Debug, Clone)]
pub struct InferencePool {
    threads: usize,
    /// `None` runs on the calling thread.
    executor: Option<Executor>,
}

impl InferencePool {
    /// A pool of `threads` threads; 0 or 1 runs every inference on the
    /// calling thread and starts no threads.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] if the threads can't be started.
    pub fn new(threads: usize) -> Result<Self> {
        if threads <= 1 {
            return Ok(Self::calling_thread());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("stance-pose-{i}"))
            .build()
            .map_err(|e| Error::model("inference pool", format!("thread pool: {e}")))?;
        Ok(Self {
            threads,
            executor: Some(Executor::MultiThread(Arc::new(pool))),
        })
    }

    /// Every inference on the calling thread.
    #[must_use]
    pub fn calling_thread() -> Self {
        Self {
            threads: 1,
            executor: None,
        }
    }

    /// The number of cores one inference may use.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }
}

impl Default for InferencePool {
    fn default() -> Self {
        Self::calling_thread()
    }
}

/// Mutable run state for one thread.
#[derive(Debug)]
pub(crate) struct Session {
    network: Network,
    state: TypedSimpleState,
    pool: InferencePool,
}

impl Session {
    /// The network this session runs.
    pub(crate) fn network(&self) -> &Network {
        &self.network
    }

    /// Runs each inference of this session on `pool`.
    pub(crate) fn set_pool(&mut self, pool: &InferencePool) {
        self.pool = pool.clone();
    }

    /// Runs the model on an NCHW input of the network's input shape.
    pub(crate) fn run(&mut self, input: &[f32]) -> Result<Vec<Output>> {
        let name = &self.network.name;
        let err = |e: tract_onnx::prelude::TractError| Error::model(name, format!("{e:#}"));
        let tensor = Tensor::from_shape(&self.network.input, input).map_err(err)?;
        let inputs = tvec!(tensor.into_tvalue());
        let state = &mut self.state;
        let outputs = match &self.pool.executor {
            Some(pool) => multithread_tract_scope(pool.clone(), || state.run(inputs)),
            None => state.run(inputs),
        }
        .map_err(err)?;
        outputs
            .iter()
            .map(|o| {
                let view = o.to_plain_array_view::<f32>().map_err(err)?;
                Ok(Output {
                    shape: view.shape().to_vec(),
                    data: view.iter().copied().collect(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_move_between_threads() {
        fn is_send<T: Send>() {}
        is_send::<Session>();
        is_send::<Network>();
        is_send::<InferencePool>();
    }

    #[test]
    fn small_pools_start_no_threads() {
        for n in [0, 1] {
            let p = InferencePool::new(n).unwrap();
            assert_eq!(p.threads(), 1);
            assert!(p.executor.is_none());
        }
        let p = InferencePool::new(2).unwrap();
        assert_eq!(p.threads(), 2);
        assert!(p.clone().executor.is_some());
    }

    #[test]
    fn garbage_is_a_model_error() {
        let e = Network::load("junk", b"not an onnx file", [1, 3, 8, 8]).unwrap_err();
        assert!(
            matches!(e, Error::Model { ref model, .. } if model == "junk"),
            "{e}"
        );
    }
}

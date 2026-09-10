pub mod backend_qbe;
pub mod backend_x64;
pub mod common;
pub mod compiler;
pub mod errors;
pub mod lexer;
pub mod optimizer;
pub mod parser;
pub mod semantic_analysis;
pub mod tacky;

pub use compiler::{compile, BackendKind, Stage};
pub use optimizer::OptimizationPasses;

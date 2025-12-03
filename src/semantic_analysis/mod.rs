mod resolve_identifiers;
mod statement_analysis;
mod type_check;
mod visitor;

pub use resolve_identifiers::resolve_variables;
pub use statement_analysis::analyze_statements;
pub use type_check::typecheck_program;

pub mod asm_ast;
pub mod backend_table;
mod emit_asm;
mod fixup;
mod platform;
mod register_allocate;
mod replace_pseudoregisters;
pub mod translate_ir;

pub use emit_asm::emit_assembly;
pub use fixup::fixup_instructions;
pub use register_allocate::allocate_program;
pub use replace_pseudoregisters::replace_pseudoregisters;

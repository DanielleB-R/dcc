pub mod asm_ast;
pub mod backend_table;
mod emit_asm;
mod fixup;
mod platform;
mod register_allocate;
mod replace_pseudoregisters;
mod translate_ir;

use emit_asm::emit_assembly;
use fixup::fixup_instructions;
use register_allocate::allocate_program;
use replace_pseudoregisters::replace_pseudoregisters;

use crate::{
    Stage,
    common::backend::Backend,
    common::{swap_suffix, write_debug_text_file},
};
use std::{fs, process};

pub struct X64Backend {
    debug: bool,
    stage: Stage,
    source_name: String,
}

impl X64Backend {
    pub fn new(debug: bool, stage: Stage, source_name: String) -> Self {
        Self {
            debug,
            stage,
            source_name,
        }
    }
}

impl Backend for X64Backend {
    fn emit(
        self,
        code: crate::tacky::ir::Program,
        symbols: crate::common::symbol_table::SymbolTable,
        types: &crate::common::type_table::TypeTable,
    ) -> Result<String, crate::errors::CompilerError> {
        let asm_program = translate_ir::translate(code, &symbols, types);

        let symbols = backend_table::convert_table(symbols, types);

        if self.debug {
            write_debug_text_file("raw_asm_ast.txt", &asm_program);
        }

        let asm_program = allocate_program(asm_program, &symbols);

        if self.debug {
            write_debug_text_file("allocated_asm_ast.txt", &asm_program);
        }

        let asm_program = fixup_instructions(replace_pseudoregisters(asm_program, &symbols));

        if self.debug {
            write_debug_text_file("processed_asm_ast.txt", &asm_program);
        }

        if self.stage == Stage::Codegen {
            process::exit(0);
        }

        let output = emit_assembly(asm_program, &symbols);

        let asm_name = swap_suffix(&self.source_name, ".c", ".s");

        fs::write(&asm_name, output)?;

        Ok(asm_name)
    }
}

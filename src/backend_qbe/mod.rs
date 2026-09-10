mod qbe_ir;
mod translate_ir;

use std::{fs, process};

use crate::common::{backend::Backend, swap_suffix};

fn qbe_process_source(ssa_name: &str) -> std::io::Result<String> {
    let asm_name = swap_suffix(ssa_name, ".ssa", ".s");

    let output = process::Command::new("qbe")
        .arg("-o")
        .arg(&asm_name)
        .arg(ssa_name)
        .output()?;

    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        process::exit(-1);
    }

    Ok(asm_name)
}

pub struct QbeBackend {
    source_name: String,
}

impl QbeBackend {
    pub fn new(source_name: String) -> Self {
        Self { source_name }
    }
}

fn emit_instruction(code: qbe_ir::Inst) -> String {
    match code {
        qbe_ir::Inst::Ret(n) => {
            format!("ret {}", n)
        }
    }
}

fn emit_ssa(code: qbe_ir::Program) -> String {
    let f = code.function;

    format!(
        "export function w ${}() {{\n@start\n{}\n}}",
        f.name.value,
        emit_instruction(f.body)
    )
}

impl Backend for QbeBackend {
    fn emit(
        self,
        code: crate::tacky::ir::Program,
        _symbols: crate::common::symbol_table::SymbolTable,
        _types: &crate::common::type_table::TypeTable,
    ) -> Result<String, crate::errors::CompilerError> {
        let qbe_program = translate_ir::translate_ir(code);

        let ssa_name = swap_suffix(&self.source_name, ".c", ".ssa");

        fs::write(&ssa_name, emit_ssa(qbe_program))?;

        Ok(qbe_process_source(&ssa_name)?)
    }
}

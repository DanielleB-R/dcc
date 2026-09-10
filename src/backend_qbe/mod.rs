use std::{fs, process};

use crate::{
    common::{backend::Backend, swap_suffix},
    tacky::ir,
};

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

fn translate_instruction(code: ir::Instruction) -> String {
    match code {
        ir::Instruction::Return(Some(ir::Value::Constant(c))) => {
            format!("ret {}", c.unwrap_integer())
        }
        _ => unimplemented!(),
    }
}

fn translate_tacky(code: ir::Program) -> String {
    match code.top_level[0].clone() {
        ir::TopLevel::Fn(f) => {
            format!(
                "export function w ${}() {{\n@start\n{}\n}}",
                f.name.value,
                translate_instruction(f.body[0].clone())
            )
        }
        _ => unimplemented!(),
    }
}

impl Backend for QbeBackend {
    fn emit(
        self,
        code: crate::tacky::ir::Program,
        _symbols: crate::common::symbol_table::SymbolTable,
        _types: &crate::common::type_table::TypeTable,
    ) -> Result<String, crate::errors::CompilerError> {
        let ssa_name = swap_suffix(&self.source_name, ".c", ".ssa");

        fs::write(&ssa_name, translate_tacky(code))?;

        Ok(qbe_process_source(&ssa_name)?)
    }
}

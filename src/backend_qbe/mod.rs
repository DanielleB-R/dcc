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

fn emit_value(code: qbe_ir::Value) -> String {
    match code {
        qbe_ir::Value::Constant(n) => format!("{}", n),
        qbe_ir::Value::Temporary(name) => format!("%{}", name),
    }
}

fn emit_binary(code: qbe_ir::BinOp) -> &'static str {
    match code {
        qbe_ir::BinOp::Add => "add",
        qbe_ir::BinOp::Sub => "sub",
        qbe_ir::BinOp::Mul => "mul",
        qbe_ir::BinOp::Div => "div",
        qbe_ir::BinOp::Rem => "rem",
        qbe_ir::BinOp::And => "and",
        qbe_ir::BinOp::Or => "or",
        qbe_ir::BinOp::Xor => "xor",
        qbe_ir::BinOp::Shl => "shl",
        qbe_ir::BinOp::Sar => "sar",
        qbe_ir::BinOp::Ceq => "ceqw",
        qbe_ir::BinOp::Cne => "cnew",
        qbe_ir::BinOp::Cslt => "csltw",
        qbe_ir::BinOp::Csle => "cslew",
        qbe_ir::BinOp::Csgt => "csgtw",
        qbe_ir::BinOp::Csge => "csgew",
    }
}

fn emit_instruction(code: qbe_ir::Inst) -> String {
    match code {
        qbe_ir::Inst::Ret(n) => {
            format!("\tret {}", emit_value(n))
        }
        qbe_ir::Inst::Negate(src, dest) => {
            format!("\t{} =w neg {}", emit_value(dest), emit_value(src),)
        }

        qbe_ir::Inst::Complement(src, dest) => {
            format!("\t{} =w xor {}, -1", emit_value(dest), emit_value(src),)
        }
        qbe_ir::Inst::Binary(op, src1, src2, dest) => {
            format!(
                "\t{} =w {} {}, {}",
                emit_value(dest),
                emit_binary(op),
                emit_value(src1),
                emit_value(src2)
            )
        }
        qbe_ir::Inst::Copy(src, dest) => {
            format!("\t{} =w copy {}", emit_value(dest), emit_value(src))
        }
        qbe_ir::Inst::Jump(label) => {
            format!("\tjmp @{}", label)
        }
        qbe_ir::Inst::Jnz(val, nz_label, z_label) => {
            format!("\tjnz {}, @{}, @{}", emit_value(val), nz_label, z_label)
        }
        qbe_ir::Inst::Label(label) => {
            format!("@{}", label)
        }
    }
}

fn emit_ssa(code: qbe_ir::Program) -> String {
    let f = code.function;

    format!(
        "export function w ${}() {{\n@start\n{}\n}}",
        f.name.value,
        f.body
            .into_iter()
            .map(emit_instruction)
            .collect::<Vec<_>>()
            .join("\n")
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

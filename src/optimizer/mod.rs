use std::collections::HashSet;

use crate::common::Identifier;
use crate::common::symbol_table::SymbolTable;
use crate::tacky::ir::{Function, Instruction, Program, TopLevel};

mod constant_folding;
pub mod control_flow;
mod copy_propagation;
mod dead_store_elimination;
mod unreachable_code_elimination;

use constant_folding::ConstantFolder;
use copy_propagation::propagate_copies;
use dead_store_elimination::eliminate_dead_stores;
use unreachable_code_elimination::eliminate_unreachable_code;

#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizationPasses {
    pub constant_folding: bool,
    pub unreachable_code_elimination: bool,
    pub copy_propagation: bool,
    pub dead_store_elimination: bool,
}

pub fn address_taken_analysis(
    instructions: &[Instruction],
    symbols: &SymbolTable,
) -> HashSet<Identifier> {
    let mut result = HashSet::new();

    for instruction in instructions {
        match instruction {
            Instruction::Copy(src, dest)
            | Instruction::SignExtend(src, dest)
            | Instruction::Truncate(src, dest)
            | Instruction::ZeroExtend(src, dest)
            | Instruction::DoubleToInt(src, dest)
            | Instruction::DoubleToUInt(src, dest)
            | Instruction::IntToDouble(src, dest)
            | Instruction::UIntToDouble(src, dest)
            | Instruction::Unary(.., src, dest)
            | Instruction::Load(src, dest)
            | Instruction::Store(src, dest) => {
                if src.is_static(symbols) {
                    result.insert(*src.unwrap_var_ref());
                }
                if dest.is_static(symbols) {
                    result.insert(*dest.unwrap_var_ref());
                }
            }
            Instruction::Binary(_, src1, src2, dest) | Instruction::AddPtr(src1, src2, _, dest) => {
                if src1.is_static(symbols) {
                    result.insert(*src1.unwrap_var_ref());
                }
                if src2.is_static(symbols) {
                    result.insert(*src2.unwrap_var_ref());
                }
                if dest.is_static(symbols) {
                    result.insert(*dest.unwrap_var_ref());
                }
            }
            Instruction::GetAddress(src, dest) => {
                result.insert(*src.unwrap_var_ref());
                if dest.is_static(symbols) {
                    result.insert(*dest.unwrap_var_ref());
                }
            }
            Instruction::CopyToOffset(val, ..)
            | Instruction::CopyFromOffset(.., val)
            | Instruction::JumpIfZero(val, _)
            | Instruction::JumpIfNotZero(val, ..)
            | Instruction::Return(Some(val)) => {
                if val.is_static(symbols) {
                    result.insert(*val.unwrap_var_ref());
                }
            }
            Instruction::FunCall(_, params, dest) => {
                for param in params {
                    if param.is_static(symbols) {
                        result.insert(*param.unwrap_var_ref());
                    }
                }
                if let Some(dest) = dest
                    && dest.is_static(symbols)
                {
                    result.insert(*dest.unwrap_var_ref());
                }
            }
            _ => {}
        }
    }

    result
}

pub fn optimize_function(
    mut function: Function,
    symbols: &SymbolTable,
    passes: OptimizationPasses,
) -> Function {
    if function.body.is_empty() {
        return function;
    }

    let constant_folder = ConstantFolder::new(symbols);
    let static_vars = symbols
        .iter()
        .filter(|(_, v)| v.attrs.is_static())
        .map(|(k, _)| -> Identifier { (*k).into() })
        .collect();

    let mut body = function.body.clone();
    loop {
        let aliased_vars = address_taken_analysis(&body, symbols);

        let post_constant_folding = if passes.constant_folding {
            constant_folder.constant_folding(body.clone())
        } else {
            body.clone()
        };

        let mut cfg = post_constant_folding.into();

        if passes.unreachable_code_elimination {
            cfg = eliminate_unreachable_code(cfg);
        }

        if passes.copy_propagation {
            cfg = propagate_copies(cfg, symbols, &aliased_vars);
        }

        if passes.dead_store_elimination {
            cfg = eliminate_dead_stores(cfg, &aliased_vars, &static_vars);
        }

        let optimized_body: Vec<_> = cfg.into();

        if optimized_body == body || optimized_body.is_empty() {
            function.body = optimized_body;
            return function;
        }

        body = optimized_body;
    }
}

pub fn optimize_program(
    code: Program,
    symbols: &SymbolTable,
    passes: OptimizationPasses,
) -> Program {
    Program {
        top_level: code
            .top_level
            .into_iter()
            .map(|t| match t {
                TopLevel::Fn(f) => optimize_function(f, symbols, passes).into(),
                t => t,
            })
            .collect(),
    }
}

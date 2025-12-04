use derive_more::{IsVariant, Unwrap};
use std::collections::HashMap;

use super::asm_ast::{AssemblyType, Register};
use crate::common::{
    CType,
    symbol_table::SymbolTable,
    type_table::{EightbyteClass, TypeTable},
};

pub fn param_registers(
    params: &[CType],
    return_in_memory: bool,
    types: &TypeTable,
) -> Vec<Register> {
    use EightbyteClass::*;
    let mut int_reg_args = if return_in_memory { 1 } else { 0 };
    let mut double_reg_args = 0;

    for c_type in params {
        if *c_type == CType::Double {
            if double_reg_args < 8 {
                double_reg_args += 1;
            }
        } else if c_type.is_scalar() {
            if int_reg_args < 6 {
                int_reg_args += 1;
            }
        } else if c_type.is_structure() {
            // we have a structure
            let struct_entry = match types.get(&c_type.unwrap_structure_ref().value) {
                Some(entry) => entry,
                None => return vec![],
            };
            let classes = struct_entry.classify(types);

            if classes[0] == MEMORY {
                continue;
            }

            let mut tentative_ints = 0;
            let mut tentative_doubles = 0;

            for class in &classes {
                if *class == SSE {
                    tentative_doubles += 1;
                } else {
                    tentative_ints += 1;
                }
            }

            if tentative_doubles + double_reg_args <= 8 && tentative_ints + int_reg_args <= 6 {
                double_reg_args += tentative_doubles;
                int_reg_args += tentative_ints;
            }
        }
    }

    let mut result = vec![];

    result.extend(Register::INT_PARAM_REGS.iter().take(int_reg_args));
    result.extend(Register::FP_PARAM_REGS.iter().take(double_reg_args));

    result
}

fn return_registers(return_type: &CType, types: &TypeTable) -> Vec<Register> {
    if *return_type == CType::Double {
        vec![Register::FP_RET_REGS[0]]
    } else if return_type.is_scalar() {
        vec![Register::INT_RET_REGS[0]]
    } else if return_type.is_structure() {
        let entry = match types.get(&return_type.unwrap_structure_ref().value) {
            Some(entry) => entry,
            None => return vec![],
        };

        let classes = entry.classify(types);
        let mut int_retvals = 0;
        let mut double_retvals = 0;

        for class in classes {
            match class {
                EightbyteClass::SSE => double_retvals += 1,
                EightbyteClass::INTEGER => int_retvals += 1,
                EightbyteClass::MEMORY => return vec![],
            }
        }
        let mut result = vec![];

        result.extend(Register::FP_RET_REGS.iter().take(double_retvals));
        result.extend(Register::INT_RET_REGS.iter().take(int_retvals));

        result
    } else {
        vec![]
    }
}

#[derive(Clone, Debug, IsVariant, Unwrap)]
#[unwrap(ref)]
pub enum BackendEntry {
    Obj(AssemblyType, bool),
    Fun(bool, bool, Vec<Register>, Vec<Register>),
}

pub type BackendTable = HashMap<&'static str, BackendEntry>;

pub fn convert_table(frontend_table: SymbolTable, type_table: &TypeTable) -> BackendTable {
    frontend_table
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                match v.c_type {
                    CType::Function(def) => {
                        let returns_in_memory = def.ret.returns_in_memory(type_table);
                        BackendEntry::Fun(
                            v.attrs.unwrap_fun().defined,
                            returns_in_memory,
                            param_registers(&def.params, returns_in_memory, type_table),
                            return_registers(&def.ret, type_table),
                        )
                    }
                    value_type => BackendEntry::Obj(
                        AssemblyType::from_c_type(value_type, type_table),
                        v.attrs.is_static() || v.attrs.is_const(),
                    ),
                },
            )
        })
        .collect()
}

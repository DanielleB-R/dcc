use std::collections::{HashMap, VecDeque};

use super::asm_ast::*;
use super::backend_table::BackendTable;
use crate::common::Identifier;

struct PseudoregisterReplacer<'a> {
    stack_posn: isize,
    pseudo_map: HashMap<&'a str, isize>,
    instructions: &'a [Inst],
    symbol_table: &'a BackendTable,
}

impl<'a> PseudoregisterReplacer<'a> {
    fn new(
        instructions: &'a [Inst],
        symbol_table: &'a BackendTable,
        return_in_memory: bool,
    ) -> Self {
        Self {
            stack_posn: if return_in_memory { -8 } else { 0 },
            pseudo_map: HashMap::new(),
            instructions,
            symbol_table,
        }
    }

    fn new_stack(&mut self, pseudo_name: &'a str) -> isize {
        let asm_type = self
            .symbol_table
            .get(pseudo_name)
            .unwrap()
            .unwrap_obj_ref()
            .0;
        let alignment = asm_type.alignment();
        if self.stack_posn % alignment != 0 {
            // stack_posn is negative so stack_posn % alignment is negative
            self.stack_posn -= alignment + self.stack_posn % alignment;
        }
        if asm_type.size() % alignment != 0 {
            self.stack_posn -= alignment - asm_type.size() % alignment;
        }
        self.stack_posn -= asm_type.size();
        self.pseudo_map.insert(pseudo_name, self.stack_posn);
        self.stack_posn
    }

    fn replace_pseudoregister(&mut self, name: &'a Identifier) -> Operand {
        if let Some(symbol) = self.symbol_table.get(&name.value)
            && *symbol.unwrap_obj_ref().1
        {
            return Operand::Data(*name, 0);
        }

        Operand::Memory(
            Register::BP,
            self.pseudo_map
                .get(&name.value)
                .copied()
                .unwrap_or_else(|| self.new_stack(name.value)),
        )
    }

    fn replace_pseudomem(&mut self, name: &'a Identifier, offset: isize) -> Operand {
        if let Some(symbol) = self.symbol_table.get(&name.value)
            && *symbol.unwrap_obj_ref().1
        {
            return Operand::Data(*name, offset as usize);
        }

        Operand::Memory(
            Register::BP,
            self.pseudo_map
                .get(&name.value)
                .copied()
                .unwrap_or_else(|| self.new_stack(name.value))
                + offset,
        )
    }

    fn replace_operand(&mut self, operand: &'a Operand) -> Operand {
        match operand {
            Operand::Pseudo(name) => self.replace_pseudoregister(name),
            Operand::PseudoMem(name, offset) => self.replace_pseudomem(name, *offset),
            o => *o,
        }
    }

    fn translate_instruction(&mut self, instruction: &'a Inst) -> Inst {
        match instruction {
            Inst::Mov(asm_type, src, dest) => mov(
                *asm_type,
                self.replace_operand(src),
                self.replace_operand(dest),
            ),
            Inst::Movsx(src_type, dest_type, src, dest) => Inst::Movsx(
                *src_type,
                *dest_type,
                self.replace_operand(src),
                self.replace_operand(dest),
            ),
            Inst::MovZeroExtend(src_type, dest_type, src, dest) => Inst::MovZeroExtend(
                *src_type,
                *dest_type,
                self.replace_operand(src),
                self.replace_operand(dest),
            ),
            Inst::Lea(src, dest) => {
                Inst::Lea(self.replace_operand(src), self.replace_operand(dest))
            }
            Inst::Cvttsd2si(dst_type, src, dest) => Inst::Cvttsd2si(
                *dst_type,
                self.replace_operand(src),
                self.replace_operand(dest),
            ),
            Inst::Cvtsi2sd(src_type, src, dest) => Inst::Cvtsi2sd(
                *src_type,
                self.replace_operand(src),
                self.replace_operand(dest),
            ),

            Inst::Unary(asm_type, operator, operand) => {
                unary(*asm_type, *operator, self.replace_operand(operand))
            }
            Inst::Binary(asm_type, operator, operand1, operand2) => binary(
                *operator,
                *asm_type,
                self.replace_operand(operand1),
                self.replace_operand(operand2),
            ),
            Inst::Idiv(asm_type, operand) => Inst::Idiv(*asm_type, self.replace_operand(operand)),
            Inst::Div(asm_type, operand) => Inst::Div(*asm_type, self.replace_operand(operand)),
            Inst::Cmp(asm_type, operand1, operand2) => Inst::Cmp(
                *asm_type,
                self.replace_operand(operand1),
                self.replace_operand(operand2),
            ),
            Inst::SetCC(condition, operand) => {
                Inst::SetCC(*condition, self.replace_operand(operand))
            }
            Inst::Push(operand) => Inst::Push(self.replace_operand(operand)),
            i => *i,
        }
    }

    fn translate_instructions(mut self) -> (VecDeque<Inst>, isize) {
        (
            self.instructions
                .iter()
                .map(|i| self.translate_instruction(i))
                .collect(),
            self.stack_posn,
        )
    }
}

fn calculate_stack_adjustment(stack_offset: isize, callee_saved_count: usize) -> usize {
    let callee_saved_bytes = 8 * callee_saved_count;
    let total_stack_bytes = callee_saved_bytes + stack_offset.unsigned_abs();
    let adjusted_stack_bytes = if total_stack_bytes.is_multiple_of(16) {
        total_stack_bytes
    } else {
        (total_stack_bytes / 16 + 1) * 16
    };
    adjusted_stack_bytes - callee_saved_bytes
}

fn replace_function_pseudoregisters(code: Function, symbol_table: &BackendTable) -> Function {
    let return_in_memory = *symbol_table
        .get(&code.name.value)
        .unwrap()
        .unwrap_fun_ref()
        .1;

    let replacer = PseudoregisterReplacer::new(&code.instructions, symbol_table, return_in_memory);

    let (mut instructions, stack_offset) = replacer.translate_instructions();

    // We'll go through this array forward for both save and restore
    // since push_front will reverse them here
    for reg in &code.callee_saved_regs {
        instructions.push_front(Inst::Push(Operand::Reg(*reg)));
    }

    let stack_adjustment = calculate_stack_adjustment(stack_offset, code.callee_saved_regs.len());

    if stack_adjustment != 0 {
        instructions.push_front(binary(
            BinOp::Sub,
            AssemblyType::Quadword,
            stack_adjustment as i64,
            Register::SP,
        ));
    }

    Function {
        name: code.name,
        global: code.global,
        stack_offset,
        instructions: instructions.into(),
        callee_saved_regs: code.callee_saved_regs,
        aliased_vars: code.aliased_vars,
    }
}

pub fn replace_pseudoregisters(code: Program, symbol_table: &BackendTable) -> Program {
    code.map_fn(|f| replace_function_pseudoregisters(f, symbol_table))
}

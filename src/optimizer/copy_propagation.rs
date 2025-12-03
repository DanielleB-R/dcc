use std::collections::HashSet;
use std::mem;

use super::control_flow::{
    Annotations, ControlFlowAnalysis, ControlFlowGraph, ControlFlowNode, NodeId, forward_algorithm,
};
use crate::common::Identifier;
use crate::common::symbol_table::SymbolTable;
use crate::tacky::ir::{Instruction, Value};

type CodeCopy = (Value, Value);
type ReachingCopies = HashSet<CodeCopy>;

struct CopyPropagator<'a> {
    symbols: &'a SymbolTable,
    aliased_vars: &'a HashSet<Identifier>,
}

impl<'a> ControlFlowAnalysis<Instruction, ReachingCopies> for CopyPropagator<'a> {
    fn transfer(
        &self,
        id: NodeId,
        instructions: &[Instruction],
        annotations: &mut Annotations<ReachingCopies>,
        initial_value: ReachingCopies,
    ) {
        let mut current_reaching_copies = initial_value;

        for (i, instruction) in instructions.iter().enumerate() {
            annotations.annotate_instruction(id, i, current_reaching_copies.clone());
            match instruction {
                Instruction::Copy(src, dest) => {
                    if current_reaching_copies.contains(&(*dest, *src)) {
                        continue;
                    }

                    current_reaching_copies
                        .retain(|(copy_src, copy_dest)| copy_src != dest && copy_dest != dest);

                    let src_type = src.get_type(self.symbols);
                    let dest_type = dest.get_type(self.symbols);
                    if src_type == dest_type || src_type.is_signed() == dest_type.is_signed() {
                        current_reaching_copies.insert((*src, *dest));
                    }
                }
                Instruction::FunCall(.., dest) => {
                    current_reaching_copies.retain(|(copy_src, copy_dest)| {
                        !copy_src.is_aliased(self.aliased_vars)
                            && !copy_dest.is_aliased(self.aliased_vars)
                            && dest.as_ref().is_none_or(|dest| copy_src != dest)
                            && dest.as_ref().is_none_or(|dest| copy_dest != dest)
                    });
                }
                Instruction::Store(..) => {
                    current_reaching_copies.retain(|(copy_src, copy_dest)| {
                        !copy_src.is_aliased(self.aliased_vars)
                            && !copy_dest.is_aliased(self.aliased_vars)
                    });
                }
                Instruction::Unary(.., dest)
                | Instruction::Binary(.., dest)
                | Instruction::SignExtend(.., dest)
                | Instruction::Truncate(.., dest)
                | Instruction::ZeroExtend(.., dest)
                | Instruction::DoubleToInt(.., dest)
                | Instruction::DoubleToUInt(.., dest)
                | Instruction::IntToDouble(.., dest)
                | Instruction::UIntToDouble(.., dest)
                | Instruction::GetAddress(.., dest)
                | Instruction::Load(.., dest)
                | Instruction::AddPtr(.., dest)
                | Instruction::CopyFromOffset(.., dest) => {
                    current_reaching_copies
                        .retain(|(copy_src, copy_dest)| copy_src != dest && copy_dest != dest);
                }
                Instruction::CopyToOffset(_, dest_name, _) => {
                    let value = Value::Var(*dest_name);
                    current_reaching_copies
                        .retain(|(copy_src, copy_dest)| *copy_src != value && *copy_dest != value);
                }
                _ => {}
            }
        }
        annotations.annotate_block(id, current_reaching_copies);
    }

    fn meet(
        &self,
        block: &ControlFlowNode<Instruction>,
        annotations: &mut Annotations<ReachingCopies>,
        initial_value: &ReachingCopies,
    ) -> ReachingCopies {
        let mut incoming_copies = initial_value.clone();

        for predecessor in block.get_predecessors().iter() {
            match predecessor {
                NodeId::Entry => return Default::default(),
                NodeId::BlockId(_) => {
                    incoming_copies = incoming_copies
                        .intersection(annotations.get_block_annotation(*predecessor))
                        .cloned()
                        .collect();
                }
                NodeId::Exit => panic!("bad graph"),
            }
        }

        incoming_copies
    }
}

impl<'a> CopyPropagator<'a> {
    fn new(symbols: &'a SymbolTable, aliased_vars: &'a HashSet<Identifier>) -> Self {
        Self {
            symbols,
            aliased_vars,
        }
    }

    fn get_all_copies(&self, cfg: &ControlFlowGraph<Instruction>) -> ReachingCopies {
        let mut result: ReachingCopies = Default::default();

        for node in cfg.nodes() {
            if let ControlFlowNode::BasicBlock { instructions, .. } = node {
                for instruction in instructions {
                    if let Instruction::Copy(src, dest) = instruction
                        && src.get_type(self.symbols).is_signed()
                            == dest.get_type(self.symbols).is_signed()
                    {
                        result.insert((*src, *dest));
                    }
                }
            }
        }
        result
    }

    fn find_reaching_copies(
        &self,
        cfg: &mut ControlFlowGraph<Instruction>,
    ) -> Annotations<ReachingCopies> {
        forward_algorithm(self, cfg, self.get_all_copies(cfg))
    }
}

fn replace_operand(operand: Value, reaching_copies: &ReachingCopies) -> Value {
    match operand {
        Value::Constant(_) => operand,
        Value::Var(_) => {
            for (copy_src, copy_dest) in reaching_copies.iter() {
                if *copy_dest == operand {
                    return *copy_src;
                }
            }
            operand
        }
    }
}

fn rewrite_instruction(
    instr: Instruction,
    reaching_copies: &ReachingCopies,
) -> Option<Instruction> {
    match instr {
        Instruction::Copy(src, dest) => {
            for (copy_src, copy_dest) in reaching_copies {
                if (*copy_src == src && *copy_dest == dest)
                    || (*copy_src == dest && *copy_dest == src)
                {
                    return None;
                }
            }
            Some(Instruction::Copy(
                replace_operand(src, reaching_copies),
                dest,
            ))
        }
        Instruction::Return(val) => Some(Instruction::Return(
            val.map(|v| replace_operand(v, reaching_copies)),
        )),
        Instruction::SignExtend(src, dest) => Some(Instruction::SignExtend(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::Truncate(src, dest) => Some(Instruction::Truncate(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::ZeroExtend(src, dest) => Some(Instruction::ZeroExtend(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::DoubleToInt(src, dest) => Some(Instruction::DoubleToInt(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::DoubleToUInt(src, dest) => Some(Instruction::DoubleToUInt(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::IntToDouble(src, dest) => Some(Instruction::IntToDouble(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::UIntToDouble(src, dest) => Some(Instruction::UIntToDouble(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::Unary(op, src, dest) => Some(Instruction::Unary(
            op,
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::Binary(op, src1, src2, dest) => Some(Instruction::Binary(
            op,
            replace_operand(src1, reaching_copies),
            replace_operand(src2, reaching_copies),
            dest,
        )),
        Instruction::Load(src, dest) => Some(Instruction::Load(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::Store(src, dest) => Some(Instruction::Store(
            replace_operand(src, reaching_copies),
            dest,
        )),
        Instruction::AddPtr(ptr, index, scale, dest) => Some(Instruction::AddPtr(
            replace_operand(ptr, reaching_copies),
            replace_operand(index, reaching_copies),
            scale,
            dest,
        )),
        Instruction::CopyToOffset(src, dest, offset) => Some(Instruction::CopyToOffset(
            replace_operand(src, reaching_copies),
            dest,
            offset,
        )),
        Instruction::CopyFromOffset(src, offset, dest) => Some(Instruction::CopyFromOffset(
            replace_operand(Value::Var(src), reaching_copies).unwrap_var(),
            offset,
            dest,
        )),
        Instruction::JumpIfZero(src, target) => Some(Instruction::JumpIfZero(
            replace_operand(src, reaching_copies),
            target,
        )),
        Instruction::JumpIfNotZero(condition, target) => Some(Instruction::JumpIfNotZero(
            replace_operand(condition, reaching_copies),
            target,
        )),
        Instruction::Jump(..) | Instruction::Label(_) | Instruction::GetAddress(..) => Some(instr),
        Instruction::FunCall(name, params, dest) => Some(Instruction::FunCall(
            name,
            params
                .into_iter()
                .map(|p| replace_operand(p, reaching_copies))
                .collect(),
            dest,
        )),
    }
}

fn rewrite_cfg(
    mut cfg: ControlFlowGraph<Instruction>,
    annotations: &Annotations<ReachingCopies>,
) -> ControlFlowGraph<Instruction> {
    for node in cfg.nodes_mut() {
        if let ControlFlowNode::BasicBlock {
            id, instructions, ..
        } = node
        {
            let old_instructions = mem::take(instructions);
            instructions.extend(old_instructions.into_iter().enumerate().filter_map(
                |(i, instr)| rewrite_instruction(instr, annotations.get_inst_annotation(*id, i)),
            ));
        }
    }

    cfg
}

pub fn propagate_copies(
    mut cfg: ControlFlowGraph<Instruction>,
    symbols: &SymbolTable,
    aliased_vars: &HashSet<Identifier>,
) -> ControlFlowGraph<Instruction> {
    let propagator = CopyPropagator::new(symbols, aliased_vars);

    let annotations = propagator.find_reaching_copies(&mut cfg);

    rewrite_cfg(cfg, &annotations)
}

use std::collections::HashSet;
use std::mem;

use super::control_flow::{
    Annotations, ControlFlowAnalysis, ControlFlowGraph, ControlFlowNode, NodeId, backward_algorithm,
};
use crate::common::Identifier;
use crate::tacky::ir::{Instruction, Value};

type LiveVariables = HashSet<Identifier>;

#[derive(Debug)]
struct DeadStoreEliminator<'a> {
    static_variables: &'a LiveVariables,
    aliased_vars: &'a HashSet<Identifier>,
}

impl<'a> ControlFlowAnalysis<Instruction, LiveVariables> for DeadStoreEliminator<'a> {
    fn transfer(
        &self,
        id: NodeId,
        instructions: &[Instruction],
        annotations: &mut Annotations<LiveVariables>,
        initial_value: LiveVariables,
    ) {
        let mut current_live_variables = initial_value;

        for (i, instruction) in instructions.iter().enumerate().rev() {
            annotations.annotate_instruction(id, i, current_live_variables.clone());

            match instruction {
                Instruction::Return(val) => {
                    if let Some(Value::Var(name)) = val {
                        current_live_variables.insert(*name);
                    }
                }
                Instruction::Truncate(src, dest)
                | Instruction::SignExtend(src, dest)
                | Instruction::ZeroExtend(src, dest)
                | Instruction::DoubleToInt(src, dest)
                | Instruction::DoubleToUInt(src, dest)
                | Instruction::IntToDouble(src, dest)
                | Instruction::UIntToDouble(src, dest)
                | Instruction::Unary(_, src, dest)
                | Instruction::Copy(src, dest) => {
                    current_live_variables.remove(dest.unwrap_var_ref());
                    if let Value::Var(name) = src {
                        current_live_variables.insert(*name);
                    }
                }
                Instruction::Binary(_, src1, src2, dest)
                | Instruction::AddPtr(src1, src2, _, dest) => {
                    current_live_variables.remove(dest.unwrap_var_ref());
                    if let Value::Var(name) = src1 {
                        current_live_variables.insert(*name);
                    }
                    if let Value::Var(name) = src2 {
                        current_live_variables.insert(*name);
                    }
                }
                Instruction::GetAddress(_, dest) => {
                    current_live_variables.remove(dest.unwrap_var_ref());
                }
                Instruction::Load(src, dest) => {
                    current_live_variables.remove(dest.unwrap_var_ref());
                    if let Value::Var(name) = src {
                        current_live_variables.insert(*name);
                    }
                    current_live_variables.extend(self.aliased_vars.iter().cloned());
                }
                Instruction::Store(src, dest) => {
                    if let Value::Var(name) = src {
                        current_live_variables.insert(*name);
                    }
                    if let Value::Var(name) = dest {
                        current_live_variables.insert(*name);
                    }
                }
                Instruction::CopyToOffset(src, _, _) => {
                    if let Value::Var(name) = src {
                        current_live_variables.insert(*name);
                    }
                }
                Instruction::CopyFromOffset(src, _, dest) => {
                    current_live_variables.remove(dest.unwrap_var_ref());
                    current_live_variables.insert(*src);
                }
                Instruction::JumpIfZero(condition, _)
                | Instruction::JumpIfNotZero(condition, _) => {
                    if let Value::Var(name) = condition {
                        current_live_variables.insert(*name);
                    }
                }
                Instruction::Jump(_) | Instruction::Label(_) => {}
                Instruction::FunCall(_, args, dest) => {
                    if let Some(Value::Var(name)) = dest {
                        current_live_variables.remove(name);
                    }
                    for arg in args {
                        if let Value::Var(name) = arg {
                            current_live_variables.insert(*name);
                        }
                    }
                    current_live_variables.extend(self.static_variables.iter().cloned());
                    current_live_variables.extend(self.aliased_vars.iter().cloned());
                }
            }
        }
        annotations.annotate_block(id, current_live_variables);
    }

    fn meet(
        &self,
        block: &ControlFlowNode<Instruction>,
        annotations: &mut Annotations<LiveVariables>,
        initial_annotation: &LiveVariables,
    ) -> LiveVariables {
        let mut live_vars: LiveVariables = initial_annotation.clone();

        for successor in block.get_successors().iter() {
            match successor {
                NodeId::Exit => live_vars.extend(self.static_variables.iter().cloned()),
                NodeId::Entry => panic!("Bad graph"),
                NodeId::BlockId(_) => {
                    let successor_live_vars = annotations.get_block_annotation(*successor);
                    live_vars.extend(successor_live_vars.iter().cloned());
                }
            }
        }

        live_vars
    }
}

impl<'a> DeadStoreEliminator<'a> {
    fn new(static_variables: &'a LiveVariables, aliased_vars: &'a HashSet<Identifier>) -> Self {
        Self {
            static_variables,
            aliased_vars,
        }
    }
}

fn is_dead_store(instr: &Instruction, live_variables: &LiveVariables) -> bool {
    match instr {
        Instruction::Truncate(_, dest)
        | Instruction::SignExtend(_, dest)
        | Instruction::ZeroExtend(_, dest)
        | Instruction::DoubleToInt(_, dest)
        | Instruction::DoubleToUInt(_, dest)
        | Instruction::IntToDouble(_, dest)
        | Instruction::UIntToDouble(_, dest)
        | Instruction::Unary(_, _, dest)
        | Instruction::Copy(_, dest)
        | Instruction::GetAddress(_, dest)
        | Instruction::Load(_, dest)
        | Instruction::Binary(.., dest)
        | Instruction::AddPtr(.., dest)
        | Instruction::CopyFromOffset(.., dest) => !live_variables.contains(dest.unwrap_var_ref()),
        Instruction::CopyToOffset(_, dest, _) => !live_variables.contains(dest),
        _ => false,
    }
}

fn clear_out_cfg(
    mut cfg: ControlFlowGraph<Instruction>,
    annotations: &Annotations<LiveVariables>,
) -> ControlFlowGraph<Instruction> {
    for node in cfg.nodes_mut() {
        if let ControlFlowNode::BasicBlock {
            id, instructions, ..
        } = node
        {
            let old_instructions = mem::take(instructions);
            instructions.extend(old_instructions.into_iter().enumerate().filter_map(
                |(i, instr)| {
                    if is_dead_store(&instr, annotations.get_inst_annotation(*id, i)) {
                        None
                    } else {
                        Some(instr)
                    }
                },
            ))
        }
    }

    cfg
}

pub fn eliminate_dead_stores(
    cfg: ControlFlowGraph<Instruction>,
    aliased_vars: &HashSet<Identifier>,
    static_vars: &HashSet<Identifier>,
) -> ControlFlowGraph<Instruction> {
    let eliminator = DeadStoreEliminator::new(static_vars, aliased_vars);

    let annotations = backward_algorithm(&eliminator, &cfg, Default::default());
    clear_out_cfg(cfg, &annotations)
}

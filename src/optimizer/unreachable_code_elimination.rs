use std::collections::{HashSet, VecDeque};

use super::control_flow::{ControlFlowGraph, NodeId};
use crate::tacky::ir::Instruction;

fn remove_unreachable_nodes(
    mut cfg: ControlFlowGraph<Instruction>,
) -> ControlFlowGraph<Instruction> {
    let mut seen = HashSet::new();
    let mut to_visit = VecDeque::new();

    to_visit.push_back(NodeId::Entry);

    while let Some(node_id) = to_visit.pop_front() {
        seen.insert(node_id);
        if node_id != NodeId::Exit {
            to_visit.extend(cfg.find_node(node_id).get_successors().difference(&seen));
        }
    }

    cfg.nodes()
        .map(|n| n.get_id())
        .collect::<HashSet<_>>()
        .difference(&seen)
        .copied()
        .for_each(|id| cfg.remove_node(id));

    cfg
}

fn remove_redundant_jumps(mut cfg: ControlFlowGraph<Instruction>) -> ControlFlowGraph<Instruction> {
    let node_ids = cfg.get_sorted_node_ids();

    for i in 0..(node_ids.len() - 1) {
        let block = cfg.find_node_mut(node_ids[i]);
        if let Some(inst) = block.last_instruction()
            && inst.is_jump()
        {
            let default_successor = node_ids[i + 1];
            let successors = block.get_successors();
            if successors.len() == 1 && successors.contains(&default_successor) {
                block.remove_last_instruction();
            }
        }
    }

    cfg
}

fn remove_unused_labels(mut cfg: ControlFlowGraph<Instruction>) -> ControlFlowGraph<Instruction> {
    let node_ids = cfg.get_sorted_node_ids();

    for i in 0..node_ids.len() {
        let block = cfg.find_node_mut(node_ids[i]);
        if let Some(Instruction::Label(_)) = block.first_instruction() {
            let default_predecessor = if i == 0 {
                NodeId::Entry
            } else {
                node_ids[i - 1]
            };

            let predecessors = block.get_predecessors();
            if predecessors.len() == 1 && predecessors.contains(&default_predecessor) {
                block.remove_first_instruction();
            }
        }
    }

    cfg
}

pub fn eliminate_unreachable_code(
    cfg: ControlFlowGraph<Instruction>,
) -> ControlFlowGraph<Instruction> {
    remove_unused_labels(remove_redundant_jumps(remove_unreachable_nodes(cfg)))
}

use std::collections::{HashMap, HashSet};
use std::mem;

use super::asm_ast::{AssemblyType, BinOp, Function, Inst, Operand, Program, Register, TopLevel};
use super::backend_table::BackendTable;
use crate::common::Identifier;
use crate::optimizer::control_flow::{
    Annotations, CFGInstruction, ControlFlowAnalysis, ControlFlowGraph, ControlFlowNode,
    InstructionKind, NodeId, backward_algorithm,
};

impl CFGInstruction for Inst {
    fn is_block_starter(&self) -> bool {
        matches!(self, Inst::Label(_))
    }

    fn is_block_ender(&self) -> bool {
        matches!(self, Inst::Jmp(_) | Inst::JmpCC(..) | Inst::Ret)
    }

    fn instruction_kind(&self) -> InstructionKind {
        use Inst::*;

        match self {
            Ret => InstructionKind::Return,
            Jmp(label) => InstructionKind::Jump(*label),
            JmpCC(_, label) => InstructionKind::ConditionalJump(*label),
            Label(label) => InstructionKind::Label(*label),
            _ => InstructionKind::Other,
        }
    }
}

struct DisjointSets(HashMap<Operand, Operand>);

impl DisjointSets {
    fn new() -> Self {
        Self(Default::default())
    }

    fn union(&mut self, x: Operand, y: Operand) {
        self.0.insert(x, y);
    }

    fn find(&self, r: &Operand) -> Operand {
        if let Some(result) = self.0.get(r) {
            self.find(result)
        } else {
            *r
        }
    }

    fn nothing_coalesced(&self) -> bool {
        self.0.is_empty()
    }
}

type RegisterMap = HashMap<Identifier, Register>;

fn get_pseudoregisters(instructions: &[Inst]) -> impl Iterator<Item = Operand> {
    instructions.iter().flat_map(|inst| {
        let mut operands = vec![];
        match inst {
            Inst::Mov(_, src, dest)
            | Inst::Movsx(_, _, src, dest)
            | Inst::MovZeroExtend(_, _, src, dest)
            | Inst::Lea(src, dest)
            | Inst::Cvttsd2si(_, src, dest)
            | Inst::Cvtsi2sd(_, src, dest)
            | Inst::Binary(_, _, src, dest)
            | Inst::Cmp(_, src, dest) => {
                if src.is_pseudo() {
                    operands.push(*src);
                }
                if dest.is_pseudo() {
                    operands.push(*dest);
                }
            }
            Inst::Unary(.., dest)
            | Inst::Idiv(_, dest)
            | Inst::Div(_, dest)
            | Inst::SetCC(_, dest)
            | Inst::Push(dest) => {
                if dest.is_pseudo() {
                    operands.push(*dest);
                }
            }
            _ => {}
        }
        operands
    })
}

fn get_int_pseudoregisters(
    instructions: &[Inst],
    symbols: &BackendTable,
    aliased_vars: &HashSet<Identifier>,
) -> Vec<InterferenceNode> {
    let operands: HashSet<Operand> = get_pseudoregisters(instructions).collect();

    operands
        .into_iter()
        .filter(|op| {
            let name = op.unwrap_pseudo_ref();
            op.get_pseudo_type(symbols) != AssemblyType::DoublePrecision
                && !aliased_vars.contains(name)
        })
        .map(InterferenceNode::new)
        .collect()
}

fn get_fp_pseudoregisters(
    instructions: &[Inst],
    symbols: &BackendTable,
    aliased_vars: &HashSet<Identifier>,
) -> Vec<InterferenceNode> {
    let operands: HashSet<Operand> = get_pseudoregisters(instructions).collect();

    operands
        .into_iter()
        .filter(|op| {
            let name = op.unwrap_pseudo_ref();
            op.get_pseudo_type(symbols) == AssemblyType::DoublePrecision
                && !aliased_vars.contains(name)
        })
        .map(InterferenceNode::new)
        .collect()
}

#[derive(Debug)]
struct InterferenceNode {
    id: Operand,
    neighbours: HashSet<Operand>,
    spill_cost: f64,
    colour: Option<usize>,
}

impl InterferenceNode {
    fn new(operand: Operand) -> Self {
        Self {
            id: operand,
            neighbours: Default::default(),
            spill_cost: 0.0,
            colour: None,
        }
    }

    fn new_complete(reg: Register, all: &[Register]) -> Self {
        Self {
            id: reg.into(),
            neighbours: all
                .iter()
                .filter(|r| **r != reg)
                .map(|r| (*r).into())
                .collect(),
            spill_cost: 0.0,
            colour: None,
        }
    }

    fn degree(&self, pruned: &HashSet<Operand>) -> usize {
        self.neighbours
            .iter()
            .filter(|o| !pruned.contains(o))
            .count()
    }

    fn spill_metric(&self, pruned: &HashSet<Operand>) -> f64 {
        self.spill_cost / (self.degree(pruned) as f64)
    }
}

fn find_used_and_updated(
    instruction: &Inst,
    symbols: &BackendTable,
) -> (Vec<Operand>, Vec<Operand>) {
    let (mut used, updated) = match instruction {
        Inst::Mov(_, src, dest)
        | Inst::Movsx(.., src, dest)
        | Inst::MovZeroExtend(.., src, dest)
        | Inst::Lea(src, dest)
        | Inst::Cvttsd2si(_, src, dest)
        | Inst::Cvtsi2sd(_, src, dest) => (vec![*src], vec![*dest]),
        Inst::Binary(_, BinOp::Shl | BinOp::Shr | BinOp::Sar, src, dest) if !src.is_imm() => (
            vec![Register::CX.into(), *src, *dest],
            vec![Register::CX.into(), *dest],
        ),
        Inst::Binary(.., src, dest) => (vec![*src, *dest], vec![*dest]),
        Inst::Unary(.., dest) => (vec![*dest], vec![*dest]),
        Inst::Cmp(_, src, dest) => (vec![*src, *dest], vec![]),
        Inst::SetCC(_, dest) => (vec![], vec![*dest]),
        Inst::Push(src) => (vec![*src], vec![]),
        Inst::Pop(dest) => (vec![], vec![Operand::Reg(*dest)]),
        Inst::Idiv(_, divisor) | Inst::Div(_, divisor) => (
            vec![*divisor, Register::AX.into(), Register::DX.into()],
            vec![Register::AX.into(), Register::DX.into()],
        ),
        Inst::Cdq(_) => (vec![Register::AX.into()], vec![Register::DX.into()]),
        Inst::Call(name) => {
            let arg_regs = symbols.get(&name.value).unwrap().unwrap_fun_ref().2.clone();
            (
                arg_regs.into_iter().map(|reg| reg.into()).collect(),
                vec![
                    Register::DI.into(),
                    Register::SI.into(),
                    Register::DX.into(),
                    Register::CX.into(),
                    Register::R8.into(),
                    Register::R9.into(),
                    Register::AX.into(),
                    Register::XMM0.into(),
                    Register::XMM1.into(),
                    Register::XMM2.into(),
                    Register::XMM3.into(),
                    Register::XMM4.into(),
                    Register::XMM5.into(),
                    Register::XMM6.into(),
                    Register::XMM7.into(),
                    Register::XMM8.into(),
                    Register::XMM9.into(),
                    Register::XMM10.into(),
                    Register::XMM11.into(),
                    Register::XMM12.into(),
                    Register::XMM13.into(),
                ],
            )
        }
        _ => (vec![], vec![]),
    };

    let mut used_from_operand = vec![];

    for op in used.iter().chain(updated.iter()) {
        match op {
            Operand::Memory(reg, _) => used_from_operand.push(Operand::Reg(*reg)),
            Operand::Indexed(reg1, reg2, _) => {
                used_from_operand.push(Operand::Reg(*reg1));
                used_from_operand.push(Operand::Reg(*reg2));
            }
            _ => {}
        }
    }

    used.extend_from_slice(&used_from_operand);
    used.sort();
    used.dedup();
    (used, updated)
}

type LiveRegisters = HashSet<Operand>;

struct LivenessAnalysis<'a> {
    symbols: &'a BackendTable,
    name: Identifier,
}

impl<'a> ControlFlowAnalysis<Inst, LiveRegisters> for LivenessAnalysis<'a> {
    fn meet(
        &self,
        block: &ControlFlowNode<Inst>,
        annotations: &mut Annotations<LiveRegisters>,
        initial_annotation: &LiveRegisters,
    ) -> LiveRegisters {
        let mut live_registers = initial_annotation.clone();
        for successor in block.get_successors() {
            match successor {
                NodeId::Exit => {
                    live_registers.extend(
                        self.symbols
                            .get(&self.name.value)
                            .unwrap()
                            .unwrap_fun_ref()
                            .3
                            .iter()
                            .map(|r| Operand::from(*r)),
                    );
                }
                NodeId::Entry => panic!("bad graph"),
                NodeId::BlockId(_) => {
                    live_registers
                        .extend(annotations.get_block_annotation(*successor).iter().cloned());
                }
            }
        }
        live_registers
    }

    fn transfer(
        &self,
        id: NodeId,
        instructions: &[Inst],
        annotations: &mut Annotations<LiveRegisters>,
        initial_value: LiveRegisters,
    ) {
        let mut current_live_registers = initial_value;

        for (i, instruction) in instructions.iter().enumerate().rev() {
            annotations.annotate_instruction(id, i, current_live_registers.clone());
            let (used, updated) = find_used_and_updated(instruction, self.symbols);

            for operand in updated {
                if operand.is_register() {
                    current_live_registers.remove(&operand);
                }
            }

            for operand in used {
                if operand.is_register() {
                    current_live_registers.insert(operand);
                }
            }
        }
        annotations.annotate_block(id, current_live_registers);
    }
}

impl<'a> LivenessAnalysis<'a> {
    fn new(symbols: &'a BackendTable, name: Identifier) -> Self {
        Self { symbols, name }
    }
}

fn add_edges(
    cfg: &ControlFlowGraph<Inst>,
    annotations: &Annotations<LiveRegisters>,
    graph: &mut InterferenceGraph,
    symbols: &BackendTable,
) {
    for node in cfg.nodes() {
        if let ControlFlowNode::BasicBlock {
            id, instructions, ..
        } = node
        {
            for (i, instr) in instructions.iter().enumerate() {
                let (_, updated) = find_used_and_updated(instr, symbols);
                let live_registers = annotations.get_inst_annotation(*id, i);

                for l in live_registers {
                    if let Inst::Mov(_, src, _) = instr
                        && l == src
                    {
                        continue;
                    }

                    if !graph.has_node_for(l) {
                        continue;
                    }

                    for u in &updated {
                        if l != u && graph.has_node_for(u) {
                            graph.add_edge(l, u);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct InterferenceGraph {
    nodes: Vec<InterferenceNode>,
}

impl InterferenceGraph {
    fn build_graph_integer(
        instructions: &[Inst],
        symbols: &BackendTable,
        aliased_vars: &HashSet<Identifier>,
    ) -> Self {
        let mut int_graph = Self {
            nodes: Register::INT_ALLOCATE_REGS
                .iter()
                .map(|r| InterferenceNode::new_complete(*r, &Register::INT_ALLOCATE_REGS))
                .collect(),
        };

        int_graph
            .nodes
            .extend(get_int_pseudoregisters(instructions, symbols, aliased_vars));

        int_graph
    }

    fn build_graph_xmm(
        instructions: &[Inst],
        symbols: &BackendTable,
        aliased_vars: &HashSet<Identifier>,
    ) -> Self {
        let mut fp_graph = Self {
            nodes: Register::FP_ALLOCATE_REGS
                .iter()
                .map(|r| InterferenceNode::new_complete(*r, &Register::FP_ALLOCATE_REGS))
                .collect(),
        };

        fp_graph
            .nodes
            .extend(get_fp_pseudoregisters(instructions, symbols, aliased_vars));

        fp_graph
    }

    fn add_spill_costs(&mut self, instructions: &[Inst]) {
        for node in &mut self.nodes {
            if node.id.is_reg() {
                node.spill_cost = f64::INFINITY;
            } else {
                node.spill_cost = get_pseudoregisters(instructions)
                    .filter(|o| *o == node.id)
                    .count() as f64;
            }
        }
    }

    fn colour_graph_ret(&mut self, k: usize, pruned: &mut HashSet<Operand>) {
        let remaining: Vec<_> = self
            .nodes
            .iter()
            .filter(|n| !pruned.contains(&n.id))
            .collect();

        if remaining.is_empty() {
            return;
        }

        let mut chosen_node_id: Option<Operand> = None;

        for node in &remaining {
            if node.degree(pruned) < k {
                chosen_node_id = Some(node.id);
                break;
            }
        }

        if chosen_node_id.is_none() {
            // time to spill
            let mut best_spill_metric = f64::INFINITY;
            for node in &remaining {
                let spill_metric = node.spill_metric(pruned);
                if spill_metric < best_spill_metric {
                    best_spill_metric = spill_metric;
                    chosen_node_id = Some(node.id);
                }
            }
        }
        let chosen_node_id = chosen_node_id.unwrap();

        pruned.insert(chosen_node_id);

        // colour the rest of the graph!
        self.colour_graph_ret(k, pruned);

        // pick the current node's colour
        let mut colours: HashSet<usize> = (1..=k).collect();
        for neighbour_id in &self.find_node(&chosen_node_id).neighbours {
            let neighbour = self.find_node(neighbour_id);
            if let Some(n) = neighbour.colour {
                colours.remove(&n);
            }
        }

        if !colours.is_empty() {
            let colour = *(if let Operand::Reg(r) = chosen_node_id
                && r.is_callee_saved()
            {
                colours.iter().max().unwrap()
            } else {
                colours.iter().min().unwrap()
            });
            self.find_node_mut(&chosen_node_id).colour = Some(colour);
            pruned.remove(&chosen_node_id);
        }
    }

    fn colour_graph(&mut self, k: usize) {
        let mut pruned = Default::default();
        self.colour_graph_ret(k, &mut pruned);
    }

    fn create_register_map(self) -> (RegisterMap, HashSet<Register>) {
        let mut colour_map = HashMap::new();
        for node in &self.nodes {
            if let Operand::Reg(r) = node.id {
                colour_map.insert(node.colour.unwrap(), r);
            }
        }

        let mut register_map = RegisterMap::new();
        let mut callee_saved_regs = HashSet::new();

        for node in self.nodes {
            if let Operand::Pseudo(name) = node.id
                && let Some(colour) = node.colour
            {
                let reg = colour_map.get(&colour).unwrap();
                register_map.insert(name, *reg);
                if reg.is_callee_saved() {
                    callee_saved_regs.insert(*reg);
                }
            }
        }

        (register_map, callee_saved_regs)
    }

    fn has_node_for(&self, operand: &Operand) -> bool {
        self.nodes.iter().any(|n| n.id == *operand)
    }

    fn find_node(&self, id: &Operand) -> &InterferenceNode {
        self.nodes.iter().find(|n| n.id == *id).unwrap()
    }

    fn find_node_mut(&mut self, id: &Operand) -> &mut InterferenceNode {
        self.nodes.iter_mut().find(|n| n.id == *id).unwrap()
    }

    fn add_edge(&mut self, a: &Operand, b: &Operand) {
        self.find_node_mut(a).neighbours.insert(*b);
        self.find_node_mut(b).neighbours.insert(*a);
    }

    fn remove_edge(&mut self, a: &Operand, b: &Operand) {
        self.find_node_mut(a).neighbours.remove(b);
        self.find_node_mut(b).neighbours.remove(a);
    }

    fn are_neighbours(&self, a: &Operand, b: &Operand) -> bool {
        self.find_node(a).neighbours.contains(b)
    }

    fn remove_node(&mut self, id: &Operand) {
        let index = self.nodes.iter().position(|n| n.id == *id).unwrap();
        self.nodes.swap_remove(index);
    }
}

fn replace_pseudoreg(operand: Operand, register_map: &RegisterMap) -> Operand {
    if let Operand::Pseudo(ref name) = operand
        && let Some(reg) = register_map.get(name)
    {
        Operand::Reg(*reg)
    } else {
        operand
    }
}

fn replace_pseudoregs(instructions: Vec<Inst>, register_map: RegisterMap) -> Vec<Inst> {
    instructions
        .into_iter()
        .filter_map(|inst| match inst {
            Inst::Mov(t, src, dest) => {
                let new_src = replace_pseudoreg(src, &register_map);
                let new_dest = replace_pseudoreg(dest, &register_map);
                if new_src == new_dest {
                    None
                } else {
                    Some(Inst::Mov(t, new_src, new_dest))
                }
            }
            Inst::Movsx(t1, t2, src, dest) => Some(Inst::Movsx(
                t1,
                t2,
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::MovZeroExtend(t1, t2, src, dest) => Some(Inst::MovZeroExtend(
                t1,
                t2,
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::Lea(src, dest) => Some(Inst::Lea(
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::Cvttsd2si(t, src, dest) => Some(Inst::Cvttsd2si(
                t,
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::Cvtsi2sd(t, src, dest) => Some(Inst::Cvtsi2sd(
                t,
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::Unary(t, op, dest) => {
                Some(Inst::Unary(t, op, replace_pseudoreg(dest, &register_map)))
            }
            Inst::Binary(t, op, src, dest) => Some(Inst::Binary(
                t,
                op,
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::Cmp(t, src, dest) => Some(Inst::Cmp(
                t,
                replace_pseudoreg(src, &register_map),
                replace_pseudoreg(dest, &register_map),
            )),
            Inst::Idiv(t, dest) => Some(Inst::Idiv(t, replace_pseudoreg(dest, &register_map))),
            Inst::Div(t, dest) => Some(Inst::Div(t, replace_pseudoreg(dest, &register_map))),
            Inst::SetCC(cc, dest) => Some(Inst::SetCC(cc, replace_pseudoreg(dest, &register_map))),
            Inst::Push(src) => Some(Inst::Push(replace_pseudoreg(src, &register_map))),
            i => Some(i),
        })
        .collect()
}

fn update_graph(graph: &mut InterferenceGraph, x: Operand, y: Operand) {
    let node_to_remove = graph.find_node(&x);
    for neighbour in node_to_remove.neighbours.clone() {
        graph.add_edge(&y, &neighbour);
        graph.remove_edge(&x, &neighbour);
    }

    graph.remove_node(&x);
}

fn conservative_coalesceable(
    graph: &mut InterferenceGraph,
    src: &Operand,
    dest: &Operand,
    k: usize,
    symbols: &BackendTable,
) -> bool {
    if src.is_pseudo()
        && dest.is_pseudo()
        && src.get_pseudo_type(symbols) != dest.get_pseudo_type(symbols)
    {
        false
    } else if briggs_test(graph, src, dest, k) {
        true
    } else if src.is_reg() {
        george_test(graph, src, dest, k)
    } else if dest.is_reg() {
        george_test(graph, dest, src, k)
    } else {
        false
    }
}

fn briggs_test(graph: &mut InterferenceGraph, x: &Operand, y: &Operand, k: usize) -> bool {
    let mut significant_neighbours = 0;

    let x_node = graph.find_node(x);
    let y_node = graph.find_node(y);

    for n in x_node.neighbours.union(&y_node.neighbours) {
        let mut degree = graph.find_node(n).neighbours.len();
        if graph.are_neighbours(n, x) && graph.are_neighbours(n, y) {
            degree -= 1;
        }
        if degree >= k {
            significant_neighbours += 1;
        }
    }

    significant_neighbours < k
}

fn george_test(
    graph: &mut InterferenceGraph,
    hardreg: &Operand,
    pseudoreg: &Operand,
    k: usize,
) -> bool {
    let pseudo_node = graph.find_node(pseudoreg);

    for n in &pseudo_node.neighbours {
        if graph.are_neighbours(n, hardreg) {
            continue;
        }
        if graph.find_node(n).neighbours.len() < k {
            continue;
        }
        return false;
    }
    true
}

fn rewrite_coalesced(instructions: Vec<Inst>, coalesced_regs: DisjointSets) -> Vec<Inst> {
    instructions
        .into_iter()
        .filter_map(|inst| match inst {
            Inst::Mov(t, src, dest) => {
                let new_src = coalesced_regs.find(&src);
                let new_dest = coalesced_regs.find(&dest);
                if new_src == new_dest {
                    None
                } else {
                    Some(Inst::Mov(t, new_src, new_dest))
                }
            }
            Inst::Movsx(t1, t2, src, dest) => Some(Inst::Movsx(
                t1,
                t2,
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::MovZeroExtend(t1, t2, src, dest) => Some(Inst::MovZeroExtend(
                t1,
                t2,
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::Lea(src, dest) => Some(Inst::Lea(
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::Cvttsd2si(t, src, dest) => Some(Inst::Cvttsd2si(
                t,
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::Cvtsi2sd(t, src, dest) => Some(Inst::Cvtsi2sd(
                t,
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::Unary(t, op, dest) => Some(Inst::Unary(t, op, coalesced_regs.find(&dest))),
            Inst::Binary(t, op, src, dest) => Some(Inst::Binary(
                t,
                op,
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::Cmp(t, src, dest) => Some(Inst::Cmp(
                t,
                coalesced_regs.find(&src),
                coalesced_regs.find(&dest),
            )),
            Inst::Idiv(t, dest) => Some(Inst::Idiv(t, coalesced_regs.find(&dest))),
            Inst::Div(t, dest) => Some(Inst::Div(t, coalesced_regs.find(&dest))),
            Inst::SetCC(cc, dest) => Some(Inst::SetCC(cc, coalesced_regs.find(&dest))),
            Inst::Push(src) => Some(Inst::Push(coalesced_regs.find(&src))),
            i => Some(i),
        })
        .collect()
}

fn coalesce(
    graph: &mut InterferenceGraph,
    instructions: &[Inst],
    k: usize,
    symbols: &BackendTable,
) -> DisjointSets {
    let mut result = DisjointSets::new();

    for inst in instructions {
        if let Inst::Mov(_, src, dest) = inst {
            let src = result.find(src);
            let dest = result.find(dest);

            if src != dest
                && graph.has_node_for(&src)
                && graph.has_node_for(&dest)
                && !graph.are_neighbours(&src, &dest)
                && conservative_coalesceable(graph, &src, &dest, k, symbols)
            {
                let (to_keep, to_merge) = if src.is_reg() {
                    (src, dest)
                } else {
                    (dest, src)
                };

                result.union(to_merge, to_keep);
                update_graph(graph, to_merge, to_keep);
            }
        }
    }

    result
}

pub fn allocate_registers(
    mut instructions: Vec<Inst>,
    name: Identifier,
    symbols: &BackendTable,
    aliased_vars: &HashSet<Identifier>,
) -> (Vec<Inst>, Vec<Register>) {
    let (mut int_graph, mut fp_graph) = loop {
        let mut int_graph =
            InterferenceGraph::build_graph_integer(&instructions, symbols, aliased_vars);
        let mut fp_graph = InterferenceGraph::build_graph_xmm(&instructions, symbols, aliased_vars);

        let cfg: ControlFlowGraph<Inst> = instructions.to_vec().into();

        let liveness_analysis = LivenessAnalysis::new(symbols, name);

        let annotations = backward_algorithm(&liveness_analysis, &cfg, Default::default());

        add_edges(&cfg, &annotations, &mut int_graph, symbols);
        add_edges(&cfg, &annotations, &mut fp_graph, symbols);

        let coalesced_int_regs = coalesce(&mut int_graph, &instructions, 12, symbols);
        let coalesced_fp_regs = coalesce(&mut fp_graph, &instructions, 14, symbols);

        if coalesced_int_regs.nothing_coalesced() && coalesced_fp_regs.nothing_coalesced() {
            break (int_graph, fp_graph);
        }

        instructions = rewrite_coalesced(instructions, coalesced_int_regs);
        instructions = rewrite_coalesced(instructions, coalesced_fp_regs);
    };

    int_graph.add_spill_costs(&instructions);
    fp_graph.add_spill_costs(&instructions);

    int_graph.colour_graph(12);
    fp_graph.colour_graph(14);

    let (int_reg_map, callee_saved_regs) = int_graph.create_register_map();
    let (fp_reg_map, _) = fp_graph.create_register_map();

    let register_map = int_reg_map.into_iter().chain(fp_reg_map).collect();

    (
        replace_pseudoregs(instructions, register_map),
        callee_saved_regs.into_iter().collect(),
    )
}

fn allocate_function(mut code: Function, symbols: &BackendTable) -> Function {
    let instructions = mem::take(&mut code.instructions);

    (code.instructions, code.callee_saved_regs) =
        allocate_registers(instructions, code.name, symbols, &code.aliased_vars);

    code
}

pub fn allocate_program(code: Program, symbols: &BackendTable) -> Program {
    Program {
        top_level: code
            .top_level
            .into_iter()
            .map(|t| match t {
                TopLevel::Fn(func) => TopLevel::Fn(allocate_function(func, symbols)),
                t => t,
            })
            .collect(),
    }
}

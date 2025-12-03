use std::collections::{HashMap, HashSet, VecDeque};

use derive_more::IsVariant;

use crate::common::CodeLabel;
use crate::tacky::ir;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, IsVariant)]
pub enum NodeId {
    Entry,
    Exit,
    BlockId(usize),
}

impl NodeId {
    fn successor(&self, max_node_id: Self) -> Self {
        match self {
            Self::Entry => Self::BlockId(0),
            Self::Exit => panic!(),
            Self::BlockId(n) if *self != max_node_id => Self::BlockId(n + 1),
            Self::BlockId(_) => Self::Exit,
        }
    }
}

pub type InstructionKey = (NodeId, usize);
pub type InstructionAnnotation<T> = HashMap<InstructionKey, T>;
pub type BlockAnnotation<T> = HashMap<NodeId, T>;

#[derive(Clone, Debug)]
pub struct Annotations<T: Clone + Eq> {
    inst_annotations: InstructionAnnotation<T>,
    block_annotations: BlockAnnotation<T>,
}

impl<T: Clone + Eq> Default for Annotations<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq> Annotations<T> {
    pub fn new() -> Self {
        Self {
            inst_annotations: Default::default(),
            block_annotations: Default::default(),
        }
    }

    pub fn annotate_block(&mut self, id: NodeId, value: T) {
        self.block_annotations.insert(id, value);
    }

    pub fn get_block_annotation(&self, id: NodeId) -> &T {
        self.block_annotations.get(&id).unwrap()
    }

    pub fn annotate_instruction(&mut self, id: NodeId, index: usize, value: T) {
        self.inst_annotations.insert((id, index), value);
    }

    pub fn get_inst_annotation(&self, id: NodeId, index: usize) -> &T {
        self.inst_annotations.get(&(id, index)).unwrap()
    }

    pub fn has_annotation_changed(&self, id: NodeId, old_annotation: &T) -> bool {
        self.get_block_annotation(id) != old_annotation
    }
}

#[derive(Clone, Debug, IsVariant)]
pub enum ControlFlowNode<T> {
    BasicBlock {
        id: NodeId,
        instructions: Vec<T>,
        predecessors: HashSet<NodeId>,
        successors: HashSet<NodeId>,
    },
    EntryNode(HashSet<NodeId>),
    ExitNode(HashSet<NodeId>),
}

impl<T> PartialEq for ControlFlowNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get_id() == other.get_id()
    }
}

impl<T> ControlFlowNode<T> {
    pub fn get_predecessors(&self) -> &HashSet<NodeId> {
        match self {
            Self::BasicBlock { predecessors, .. } => predecessors,
            Self::ExitNode(p) => p,
            Self::EntryNode(_) => panic!(),
        }
    }

    pub fn get_predecessors_mut(&mut self) -> &mut HashSet<NodeId> {
        match self {
            Self::BasicBlock { predecessors, .. } => predecessors,
            Self::ExitNode(p) => p,
            Self::EntryNode(_) => panic!(),
        }
    }

    fn add_predecessor(&mut self, id: NodeId) {
        self.get_predecessors_mut().insert(id);
    }

    pub fn get_successors(&self) -> &HashSet<NodeId> {
        match self {
            Self::BasicBlock { successors, .. } => successors,
            Self::EntryNode(s) => s,
            Self::ExitNode(_) => panic!(),
        }
    }

    pub fn get_successors_mut(&mut self) -> &mut HashSet<NodeId> {
        match self {
            Self::BasicBlock { successors, .. } => successors,
            Self::EntryNode(s) => s,
            Self::ExitNode(_) => panic!(),
        }
    }

    fn add_successor(&mut self, id: NodeId) {
        self.get_successors_mut().insert(id);
    }

    pub fn get_id(&self) -> NodeId {
        match self {
            Self::BasicBlock { id, .. } => *id,
            Self::EntryNode(_) => NodeId::Entry,
            Self::ExitNode(_) => NodeId::Exit,
        }
    }

    fn remove_edges(&mut self, id: NodeId) {
        match self {
            Self::BasicBlock {
                predecessors,
                successors,
                ..
            } => {
                predecessors.remove(&id);
                successors.remove(&id);
            }
            Self::EntryNode(s) => {
                s.remove(&id);
            }
            Self::ExitNode(p) => {
                p.remove(&id);
            }
        }
    }

    pub fn first_instruction(&self) -> Option<&T> {
        match self {
            Self::BasicBlock { instructions, .. } => instructions.first(),
            _ => None,
        }
    }

    pub fn last_instruction(&self) -> Option<&T> {
        match self {
            Self::BasicBlock { instructions, .. } => instructions.last(),
            _ => None,
        }
    }

    pub fn remove_first_instruction(&mut self) {
        if let Self::BasicBlock { instructions, .. } = self {
            instructions.remove(0);
        }
    }
    pub fn remove_last_instruction(&mut self) {
        if let Self::BasicBlock { instructions, .. } = self {
            instructions.pop();
        }
    }
}

impl<T> From<ControlFlowNode<T>> for Vec<T> {
    fn from(value: ControlFlowNode<T>) -> Self {
        match value {
            ControlFlowNode::BasicBlock { instructions, .. } => instructions,
            _ => vec![],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum InstructionKind {
    Return,
    Jump(CodeLabel),
    ConditionalJump(CodeLabel),
    Label(CodeLabel),
    Other,
}

pub trait CFGInstruction {
    fn is_block_starter(&self) -> bool;
    fn is_block_ender(&self) -> bool;

    fn instruction_kind(&self) -> InstructionKind;
}

impl CFGInstruction for ir::Instruction {
    fn is_block_starter(&self) -> bool {
        matches!(self, ir::Instruction::Label(_))
    }

    fn is_block_ender(&self) -> bool {
        matches!(
            self,
            ir::Instruction::Jump(_)
                | ir::Instruction::JumpIfZero(..)
                | ir::Instruction::JumpIfNotZero { .. }
                | ir::Instruction::Return(_)
        )
    }

    fn instruction_kind(&self) -> InstructionKind {
        use ir::Instruction::*;
        match self {
            Return(_) => InstructionKind::Return,
            Jump(label) => InstructionKind::Jump(*label),
            JumpIfZero(_, target) | JumpIfNotZero(_, target) => {
                InstructionKind::ConditionalJump(*target)
            }
            Label(label) => InstructionKind::Label(*label),
            _ => InstructionKind::Other,
        }
    }
}

pub struct ControlFlowGraph<T: CFGInstruction>(HashMap<NodeId, ControlFlowNode<T>>);

impl<T: CFGInstruction> Default for ControlFlowGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CFGInstruction> ControlFlowGraph<T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn nodes(&self) -> impl Iterator<Item = &ControlFlowNode<T>> {
        self.0.values()
    }

    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut ControlFlowNode<T>> {
        self.0.values_mut()
    }

    fn partition_into_basic_blocks(instructions: Vec<T>) -> Vec<Vec<T>> {
        let mut finished_blocks = vec![];
        let mut current_block = vec![];

        for instruction in instructions {
            if instruction.is_block_starter() {
                if !current_block.is_empty() {
                    finished_blocks.push(current_block);
                }
                current_block = vec![instruction];
            } else if instruction.is_block_ender() {
                current_block.push(instruction);
                finished_blocks.push(current_block);
                current_block = vec![];
            } else {
                current_block.push(instruction);
            }
        }

        if !current_block.is_empty() {
            finished_blocks.push(current_block);
        }

        finished_blocks
    }

    pub fn find_node(&self, id: NodeId) -> &ControlFlowNode<T> {
        self.0.get(&id).unwrap()
    }

    pub fn find_node_mut(&mut self, id: NodeId) -> &mut ControlFlowNode<T> {
        self.0.get_mut(&id).unwrap()
    }

    fn add_edge(&mut self, start: NodeId, end: NodeId) {
        self.find_node_mut(start).add_successor(end);
        self.find_node_mut(end).add_predecessor(start);
    }

    fn block_id_map(&self) -> HashMap<CodeLabel, NodeId> {
        let mut result = HashMap::new();

        for block in self.0.values() {
            if let ControlFlowNode::BasicBlock {
                id, instructions, ..
            } = block
                && let Some(InstructionKind::Label(label)) =
                    instructions.first().map(|i| i.instruction_kind())
            {
                result.insert(label, *id);
            }
        }

        result
    }

    fn add_all_edges(&mut self) {
        use InstructionKind::*;
        use NodeId::*;
        self.add_edge(Entry, BlockId(0));

        let max_node_id = *self.0.keys().max().unwrap();
        let label_map = self.block_id_map();

        let edges = self
            .0
            .values()
            .flat_map(|node| {
                if let ControlFlowNode::BasicBlock {
                    id, instructions, ..
                } = node
                {
                    let next_id = id.successor(max_node_id);

                    let last_instruction = instructions.last().unwrap();

                    match last_instruction.instruction_kind() {
                        Return => vec![(*id, NodeId::Exit)],
                        Jump(target) => {
                            let target_id = label_map.get(&target).unwrap();
                            vec![(*id, *target_id)]
                        }
                        ConditionalJump(target) => {
                            let target_id = label_map.get(&target).unwrap();
                            vec![(*id, *target_id), (*id, next_id)]
                        }
                        _ => vec![(*id, next_id)],
                    }
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();

        for (from, to) in edges {
            self.add_edge(from, to);
        }
    }

    pub fn remove_node(&mut self, node_id: NodeId) {
        self.0.remove(&node_id);

        self.0.values_mut().for_each(|n| n.remove_edges(node_id));
    }

    pub fn get_sorted_node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<_> = self.0.keys().copied().filter(NodeId::is_block_id).collect();
        ids.sort();
        ids
    }
}

impl<T: CFGInstruction> From<Vec<T>> for ControlFlowGraph<T> {
    fn from(value: Vec<T>) -> Self {
        let blocks = Self::partition_into_basic_blocks(value);
        let mut graph: Self = blocks.into();
        graph.add_all_edges();
        graph
    }
}

impl<T: CFGInstruction> From<Vec<Vec<T>>> for ControlFlowGraph<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let mut nodes = HashMap::new();

        nodes.insert(NodeId::Entry, ControlFlowNode::EntryNode(HashSet::new()));

        for (block_id, block) in value.into_iter().enumerate() {
            let id = NodeId::BlockId(block_id);
            nodes.insert(
                id,
                ControlFlowNode::BasicBlock {
                    id,
                    instructions: block,
                    predecessors: HashSet::new(),
                    successors: HashSet::new(),
                },
            );
        }

        nodes.insert(NodeId::Exit, ControlFlowNode::ExitNode(HashSet::new()));

        Self(nodes)
    }
}

impl<T: CFGInstruction> From<ControlFlowGraph<T>> for Vec<T> {
    fn from(value: ControlFlowGraph<T>) -> Self {
        let mut nodes: Vec<_> = value.0.into_values().collect();
        nodes.sort_by_key(|n| n.get_id());
        nodes
            .into_iter()
            .flat_map(|n| -> Vec<T> { n.into() })
            .collect()
    }
}

pub trait ControlFlowAnalysis<T, A>
where
    A: Clone + Eq,
{
    fn transfer(
        &self,
        id: NodeId,
        instructions: &[T],
        annotations: &mut Annotations<A>,
        initial_value: A,
    );

    fn meet(
        &self,
        block: &ControlFlowNode<T>,
        annotations: &mut Annotations<A>,
        initial_annotation: &A,
    ) -> A;
}

pub fn forward_algorithm<T: CFGInstruction, A: Clone + Eq>(
    algo: &impl ControlFlowAnalysis<T, A>,
    cfg: &ControlFlowGraph<T>,
    initial_annotation: A,
) -> Annotations<A> {
    let mut annotations = Annotations::new();
    let mut worklist = VecDeque::new();

    for node in cfg.nodes() {
        if let ControlFlowNode::BasicBlock { id, .. } = node {
            worklist.push_back(node);
            annotations.annotate_block(*id, initial_annotation.clone());
        }
    }

    while let Some(block) = worklist.pop_front() {
        let old_annotation = annotations.get_block_annotation(block.get_id()).clone();
        let incoming = algo.meet(block, &mut annotations, &initial_annotation);
        if let ControlFlowNode::BasicBlock {
            id,
            instructions,
            successors,
            ..
        } = block
        {
            algo.transfer(*id, instructions, &mut annotations, incoming);
            if !annotations.has_annotation_changed(*id, &old_annotation) {
                continue;
            }

            for successor in successors {
                match successor {
                    NodeId::Entry => panic!("bad graph"),
                    NodeId::Exit => {
                        continue;
                    }
                    NodeId::BlockId(_) => {
                        let successor_block = cfg.find_node(*successor);
                        if !worklist.contains(&successor_block) {
                            worklist.push_back(successor_block);
                        }
                    }
                }
            }
        }
    }

    annotations
}

pub fn backward_algorithm<T: CFGInstruction, A: Clone + Eq>(
    algo: &impl ControlFlowAnalysis<T, A>,
    cfg: &ControlFlowGraph<T>,
    initial_annotation: A,
) -> Annotations<A> {
    let mut annotations = Annotations::new();
    let mut worklist = VecDeque::new();

    for node in cfg.nodes() {
        if let ControlFlowNode::BasicBlock { id, .. } = node {
            worklist.push_front(node);
            annotations.annotate_block(*id, initial_annotation.clone());
        }
    }

    while let Some(block) = worklist.pop_front() {
        let old_annotation = annotations.get_block_annotation(block.get_id()).clone();
        let incoming_vars = algo.meet(block, &mut annotations, &initial_annotation);

        if let ControlFlowNode::BasicBlock {
            id,
            instructions,
            predecessors,
            ..
        } = block
        {
            algo.transfer(*id, instructions, &mut annotations, incoming_vars);
            if annotations.has_annotation_changed(*id, &old_annotation) {
                for predecessor in predecessors {
                    match predecessor {
                        NodeId::Entry => {
                            continue;
                        }
                        NodeId::Exit => panic!("bad graph"),
                        NodeId::BlockId(_) => {
                            let predecessor_block = cfg.find_node(*predecessor);
                            if !worklist.contains(&predecessor_block) {
                                worklist.push_back(predecessor_block);
                            }
                        }
                    }
                }
            }
        }
    }
    annotations
}

use crate::common::symbol_table::SymbolTable;
use crate::tacky::ir::{BinaryOp, Instruction, UnaryOperator, Value};

pub struct ConstantFolder<'a> {
    symbols: &'a SymbolTable,
}

impl<'a> ConstantFolder<'a> {
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self { symbols }
    }

    fn fold_instruction(&self, inst: Instruction) -> Option<Instruction> {
        match inst {
            Instruction::Unary(op, Value::Constant(c), dest) => {
                let target_type = &dest.get_type(self.symbols);
                let new_const = match op {
                    UnaryOperator::Negate => -c.convert_type(target_type),
                    UnaryOperator::Complement => !c.convert_type(target_type),
                    UnaryOperator::Not => c.logical_not(),
                    _ => panic!("no increments in tacky"),
                };

                Some(Instruction::Copy(new_const.into(), dest))
            }
            Instruction::Binary(op, c1 @ Value::Constant(_), Value::Constant(c2), dest) => {
                let target_type = &c1.get_type(self.symbols);
                let c1_typed = c1.unwrap_constant().convert_type(target_type);
                let c2_typed = c2.convert_type(target_type);

                let new_const = match op {
                    BinaryOp::Add => c1_typed + c2_typed,
                    BinaryOp::Subtract => c1_typed - c2_typed,
                    BinaryOp::Multiply => c1_typed * c2_typed,
                    BinaryOp::Divide => c1_typed / c2_typed,
                    BinaryOp::Remainder => c1_typed % c2_typed,
                    BinaryOp::BitwiseAnd => c1_typed & c2_typed,
                    BinaryOp::BitwiseOr => c1_typed | c2_typed,
                    BinaryOp::BitwiseXor => c1_typed ^ c2_typed,
                    BinaryOp::Equal => c1_typed.c_equals(&c2_typed).into(),
                    BinaryOp::NotEqual => (!c1_typed.c_equals(&c2_typed)).into(),
                    BinaryOp::LeftShift => c1_typed << c2_typed,
                    BinaryOp::RightShift => c1_typed >> c2_typed,
                    BinaryOp::LessThan => (c1_typed < c2_typed).into(),
                    BinaryOp::LessOrEqual => (c1_typed <= c2_typed).into(),
                    BinaryOp::GreaterThan => (c1_typed > c2_typed).into(),
                    BinaryOp::GreaterOrEqual => (c1_typed >= c2_typed).into(),
                };
                Some(Instruction::Copy(new_const.into(), dest))
            }

            Instruction::JumpIfZero(Value::Constant(c), target) => {
                if c.is_zero() {
                    Some(Instruction::Jump(target))
                } else {
                    None
                }
            }
            Instruction::JumpIfNotZero(Value::Constant(c), target) => {
                if c.is_zero() {
                    None
                } else {
                    Some(Instruction::Jump(target))
                }
            }
            Instruction::Copy(Value::Constant(c), dest)
            | Instruction::Truncate(Value::Constant(c), dest)
            | Instruction::SignExtend(Value::Constant(c), dest)
            | Instruction::DoubleToInt(Value::Constant(c), dest)
            | Instruction::DoubleToUInt(Value::Constant(c), dest)
            | Instruction::IntToDouble(Value::Constant(c), dest)
            | Instruction::UIntToDouble(Value::Constant(c), dest)
            | Instruction::ZeroExtend(Value::Constant(c), dest) => {
                let target_type = &dest.get_type(self.symbols);

                Some(Instruction::Copy(c.convert_type(target_type).into(), dest))
            }
            i => Some(i),
        }
    }

    pub fn constant_folding(&self, body: Vec<Instruction>) -> Vec<Instruction> {
        body.into_iter()
            .filter_map(|i| self.fold_instruction(i))
            .collect()
    }
}

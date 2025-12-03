use std::mem;

use super::ir::{self, BinaryOp, ExpressionResult, Instruction, Value};
use crate::common::symbol_table::{
    IdentifierAttrs, InitialValue, StaticAttr, StaticInit, SymbolEntry, SymbolTable,
};
use crate::common::{CType, CodeLabel, Constant, Identifier, type_table::TypeTable};
use crate::parser::ast::{self, *};

struct Tackier<'a> {
    temporary_index: usize,
    label_index: usize,
    constant_index: usize,
    instructions: Vec<Instruction>,
    symbol_table: &'a mut SymbolTable,
    type_table: &'a TypeTable,
}

static BREAK_TAG: &str = ".break_loop";
static CONTINUE_TAG: &str = ".continue_loop";

fn convert_loop_label(tag: &'static str, loop_label: CodeLabel) -> CodeLabel {
    CodeLabel {
        tag,
        counter: loop_label.counter,
    }
}

fn chunk_string_init(s: &str, target_len: usize) -> Vec<Constant> {
    let mut result = vec![];
    let mut bytes = s.as_bytes();
    let mut remaining_len = target_len;

    while !bytes.is_empty() {
        if bytes.len() >= 8 {
            result.push(Constant::Long(i64::from_le_bytes(
                bytes[..8].try_into().unwrap(),
            )));
            bytes = &bytes[8..];
            remaining_len -= 8;
        } else if bytes.len() >= 4 {
            result.push(Constant::Int(i32::from_le_bytes(
                bytes[..4].try_into().unwrap(),
            )));
            bytes = &bytes[4..];
            remaining_len -= 4;
        } else {
            result.push(Constant::UChar(bytes[0]));
            bytes = &bytes[1..];
            remaining_len -= 1;
        }
    }

    while remaining_len > 0 {
        result.push(Constant::Char(0));
        remaining_len -= 1;
    }

    result
}

impl<'a> Tackier<'a> {
    fn new(symbol_table: &'a mut SymbolTable, type_table: &'a TypeTable) -> Self {
        Self {
            temporary_index: 0,
            label_index: 0,
            constant_index: 0,
            instructions: vec![],
            symbol_table,
            type_table,
        }
    }

    fn emit(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    fn make_temporary(&mut self, target_type: &CType) -> Value {
        self.temporary_index += 1;

        let name = format!(".tmp.{}", self.temporary_index).leak();

        self.symbol_table.insert(
            name,
            SymbolEntry {
                c_type: target_type.clone(),
                attrs: IdentifierAttrs::Local,
            },
        );

        Identifier {
            value: name,
            line: 0,
            location: 0,
        }
        .into()
    }

    fn make_string_constant(&mut self, s: String, value_type: CType) -> Value {
        self.constant_index += 1;

        let name = format!("..string.{}", self.constant_index).leak();

        if value_type.size(self.type_table) == s.len() {
            self.symbol_table.insert(
                name,
                SymbolEntry {
                    c_type: CType::Array(Box::from(CType::Char), s.len()),
                    attrs: IdentifierAttrs::Const(StaticInit::StringInit(s, false)),
                },
            );
        } else {
            self.symbol_table.insert(
                name,
                SymbolEntry {
                    c_type: CType::Array(Box::from(CType::Char), s.len() + 1),
                    attrs: IdentifierAttrs::Const(StaticInit::StringInit(s, true)),
                },
            );
        }

        Identifier {
            value: name,
            line: 0,
            location: 0,
        }
        .into()
    }

    fn make_label(&mut self, tag: &'static str) -> CodeLabel {
        self.label_index += 1;
        CodeLabel {
            tag,
            counter: self.label_index,
        }
    }

    fn emit_postfix(
        &mut self,
        operator: ast::PostfixOperator,
        inner: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let lvalue = self.emit_tacky_exp(inner);

        let operand = match lvalue {
            ExpressionResult::PlainOperand(operand) => operand,
            ExpressionResult::DereferencedPointer(ptr) => {
                let operand_intermediate = self.make_temporary(&value_type);
                self.emit(Instruction::Load(ptr, operand_intermediate));
                operand_intermediate
            }
            ExpressionResult::SubObject(base, offset) => {
                let operand_intermediate = self.make_temporary(&value_type);
                self.emit(Instruction::CopyFromOffset(
                    base,
                    offset,
                    operand_intermediate,
                ));
                operand_intermediate
            }
        };

        let dest = self.make_temporary(&value_type);

        self.emit(Instruction::Copy(operand, dest));

        if value_type.is_pointer() {
            self.emit(Instruction::AddPtr(
                operand,
                if operator == ast::PostfixOperator::Increment {
                    Constant::Long(1).into()
                } else {
                    Constant::Long(-1).into()
                },
                value_type.referent_size(self.type_table),
                operand,
            ))
        } else {
            self.emit(Instruction::Binary(
                if operator == ast::PostfixOperator::Increment {
                    ir::BinaryOp::Add
                } else {
                    ir::BinaryOp::Subtract
                },
                operand,
                Constant::one(&value_type, self.type_table).into(),
                operand,
            ));
        }

        match lvalue {
            ExpressionResult::DereferencedPointer(ptr) => {
                self.emit(Instruction::Store(operand, ptr));
            }
            ExpressionResult::SubObject(base, offset) => {
                self.emit(Instruction::CopyToOffset(operand, base, offset))
            }
            _ => {}
        }

        dest.into()
    }

    fn emit_pre_increment(
        &mut self,
        operator: ast::UnaryOperator,
        inner: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let lvalue = self.emit_tacky_exp(inner);

        let operand = match lvalue {
            ExpressionResult::PlainOperand(operand) => operand,
            ExpressionResult::DereferencedPointer(ptr) => {
                let operand_intermediate = self.make_temporary(&value_type);
                self.emit(Instruction::Load(ptr, operand_intermediate));
                operand_intermediate
            }
            ExpressionResult::SubObject(base, offset) => {
                let operand_intermediate = self.make_temporary(&value_type);
                self.emit(Instruction::CopyFromOffset(
                    base,
                    offset,
                    operand_intermediate,
                ));
                operand_intermediate
            }
        };

        self.emit(Instruction::Binary(
            if operator == ast::UnaryOperator::PreIncrement {
                ir::BinaryOp::Add
            } else {
                ir::BinaryOp::Subtract
            },
            operand,
            Constant::one(&value_type, self.type_table).into(),
            operand,
        ));

        match lvalue {
            ExpressionResult::DereferencedPointer(ptr) => {
                self.emit(Instruction::Store(operand, ptr));
            }
            ExpressionResult::SubObject(base, offset) => {
                self.emit(Instruction::CopyToOffset(operand, base, offset))
            }
            _ => {}
        }

        let dest = self.make_temporary(&value_type);
        self.emit(Instruction::Copy(operand, dest));

        dest.into()
    }

    fn emit_unary(
        &mut self,
        operator: ast::UnaryOperator,
        inner: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let src = self.emit_tacky_exp_and_convert(inner);
        let dest = self.make_temporary(&value_type);
        self.emit(Instruction::Unary(operator, src, dest));
        dest.into()
    }

    fn emit_and(&mut self, left: Expression, right: Expression) -> ExpressionResult {
        let false_label = self.make_label(".and_false");
        let end_label = self.make_label(".and_end");

        let left_condition = self.emit_tacky_exp_and_convert(left);
        self.emit(Instruction::JumpIfZero(left_condition, false_label));
        let right_condition = self.emit_tacky_exp_and_convert(right);
        self.emit(Instruction::JumpIfZero(right_condition, false_label));

        let result = self.make_temporary(&CType::Int);
        self.emit(Instruction::Copy(Constant::ONE.into(), result));
        self.emit(Instruction::Jump(end_label));
        self.emit(Instruction::Label(false_label));
        self.emit(Instruction::Copy(Constant::ZERO.into(), result));
        self.emit(Instruction::Label(end_label));
        result.into()
    }

    fn emit_or(&mut self, left: Expression, right: Expression) -> ExpressionResult {
        let true_label = self.make_label(".or_true");
        let end_label = self.make_label(".or_end");

        let left_condition = self.emit_tacky_exp_and_convert(left);
        self.emit(Instruction::JumpIfNotZero(left_condition, true_label));
        let right_condition = self.emit_tacky_exp_and_convert(right);
        self.emit(Instruction::JumpIfNotZero(right_condition, true_label));

        let result = self.make_temporary(&CType::Int);
        self.emit(Instruction::Copy(Constant::ZERO.into(), result));
        self.emit(Instruction::Jump(end_label));
        self.emit(Instruction::Label(true_label));
        self.emit(Instruction::Copy(Constant::ONE.into(), result));
        self.emit(Instruction::Label(end_label));
        result.into()
    }

    fn emit_addition(
        &mut self,
        left: Expression,
        right: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let left_type = left.get_type().clone();

        let src1 = self.emit_tacky_exp_and_convert(left);
        let src2 = self.emit_tacky_exp_and_convert(right);
        let dst = self.make_temporary(&value_type);

        if value_type.is_pointer() {
            if left_type.is_pointer() {
                self.emit(Instruction::AddPtr(
                    src1,
                    src2,
                    value_type.referent_size(self.type_table),
                    dst,
                ));
            } else {
                self.emit(Instruction::AddPtr(
                    src2,
                    src1,
                    value_type.referent_size(self.type_table),
                    dst,
                ));
            }
        } else {
            self.emit(Instruction::Binary(BinaryOp::Add, src1, src2, dst));
        }
        dst.into()
    }

    fn emit_subtraction(
        &mut self,
        left: Expression,
        right: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let left_type = left.get_type().clone();
        let right_type = right.get_type().clone();

        let src1 = self.emit_tacky_exp_and_convert(left);
        let src2 = self.emit_tacky_exp_and_convert(right);
        let dst = self.make_temporary(&value_type);

        if left_type.is_pointer() {
            if right_type.is_pointer() {
                let diff = self.make_temporary(&value_type);
                self.emit(Instruction::Binary(BinaryOp::Subtract, src1, src2, diff));

                self.emit(Instruction::Binary(
                    BinaryOp::Divide,
                    diff,
                    Value::Constant(Constant::Long(
                        left_type.referent_size(self.type_table) as i64
                    )),
                    dst,
                ));
            } else {
                let negated = self.make_temporary(&right_type);
                self.emit(Instruction::Unary(UnaryOperator::Negate, src2, negated));

                self.emit(Instruction::AddPtr(
                    src1,
                    negated,
                    value_type.referent_size(self.type_table),
                    dst,
                ));
            }
        } else {
            self.emit(Instruction::Binary(BinaryOp::Subtract, src1, src2, dst));
        }
        dst.into()
    }

    fn emit_binary(
        &mut self,
        operator: BinaryOperator,
        left: Expression,
        right: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let src1 = self.emit_tacky_exp_and_convert(left);
        let src2 = self.emit_tacky_exp_and_convert(right);
        let dst = self.make_temporary(&value_type);

        self.emit(Instruction::Binary(
            operator.try_into().unwrap(),
            src1,
            src2,
            dst,
        ));
        dst.into()
    }

    fn emit_assignment(&mut self, lhs: ast::Expression, rhs: Expression) -> ExpressionResult {
        let target = self.emit_tacky_exp(lhs);
        let result = self.emit_tacky_exp_and_convert(rhs);
        match target {
            ExpressionResult::PlainOperand(val) => {
                self.emit(Instruction::Copy(result, val));
                val.into()
            }
            ExpressionResult::DereferencedPointer(ptr) => {
                self.emit(Instruction::Store(result, ptr));
                result.into()
            }
            ExpressionResult::SubObject(base, offset) => {
                self.emit(Instruction::CopyToOffset(result, base, offset));
                result.into()
            }
        }
    }

    fn emit_compound_assignment(
        &mut self,
        op: BinaryOperator,
        left: Expression,
        right: Expression,
        target_type: Option<CType>,
        value_type: CType,
    ) -> ExpressionResult {
        let intermediate_type = target_type.unwrap_or_else(|| left.get_type().clone());

        let left_result = self.emit_tacky_exp(left);
        let left_operand = match left_result {
            ExpressionResult::PlainOperand(operand) => operand,
            ExpressionResult::DereferencedPointer(ptr) => {
                let operand_intermediate = self.make_temporary(&value_type);
                self.emit(Instruction::Load(ptr, operand_intermediate));
                operand_intermediate
            }
            ExpressionResult::SubObject(base, offset) => {
                let operand_intermediate = self.make_temporary(&value_type);
                self.emit(Instruction::CopyFromOffset(
                    base,
                    offset,
                    operand_intermediate,
                ));
                operand_intermediate
            }
        };

        let converted_left = if intermediate_type != value_type {
            let conversion_intermediate = self.make_temporary(&intermediate_type);
            self.emit_type_conversion(
                &value_type,
                &intermediate_type,
                left_operand,
                conversion_intermediate,
            );
            conversion_intermediate
        } else {
            left_operand
        };

        let right_result = self.emit_tacky_exp_and_convert(right);
        let result = self.make_temporary(&intermediate_type);
        if value_type.is_pointer() {
            if op == BinaryOperator::Add {
                self.emit(Instruction::AddPtr(
                    converted_left,
                    right_result,
                    intermediate_type.referent_size(self.type_table),
                    result,
                ));
            } else if op == BinaryOperator::Subtract {
                let right_negated = self.make_temporary(&intermediate_type);
                self.emit(Instruction::Unary(
                    UnaryOperator::Negate,
                    right_result,
                    right_negated,
                ));
                self.emit(Instruction::AddPtr(
                    converted_left,
                    right_negated,
                    intermediate_type.referent_size(self.type_table),
                    result,
                ))
            } else {
                panic!("unexpected pointer operator");
            }
        } else {
            self.emit(Instruction::Binary(
                op.try_into().unwrap(),
                converted_left,
                right_result,
                result,
            ));
        }

        let target = match &left_result {
            ExpressionResult::PlainOperand(_) => left_operand,
            ExpressionResult::DereferencedPointer(_) | ExpressionResult::SubObject(..) => {
                self.make_temporary(&value_type)
            }
        };

        self.emit_type_conversion(&intermediate_type, &value_type, result, target);

        match left_result {
            ExpressionResult::PlainOperand(_) => {}
            ExpressionResult::DereferencedPointer(ptr) => {
                self.emit(Instruction::Store(target, ptr))
            }
            ExpressionResult::SubObject(base, offset) => {
                self.emit(Instruction::CopyToOffset(target, base, offset));
            }
        }

        target.into()
    }

    fn emit_conditional(
        &mut self,
        condition: Expression,
        then_expr: Expression,
        else_expr: Expression,
        value_type: CType,
    ) -> ExpressionResult {
        let e2_label = self.make_label(".ternary_e2");
        let end_label = self.make_label(".ternary_end");

        let condition_type = condition.get_type().clone();
        let condition_value = self.emit_tacky_exp_and_convert(condition);
        self.emit(Instruction::JumpIfZero(condition_value, e2_label));
        if condition_type == CType::Void {
            self.emit_tacky_exp_and_convert(then_expr);
            self.emit(Instruction::Jump(end_label));
            self.emit(Instruction::Label(e2_label));
            self.emit_tacky_exp_and_convert(else_expr);
            self.emit(Instruction::Label(end_label));
            Value::Constant(Constant::Int(0))
        } else {
            let result = self.make_temporary(&value_type);
            let e1_result = self.emit_tacky_exp_and_convert(then_expr);
            self.emit(Instruction::Copy(e1_result, result));
            self.emit(Instruction::Jump(end_label));
            self.emit(Instruction::Label(e2_label));
            let e2_result = self.emit_tacky_exp_and_convert(else_expr);
            self.emit(Instruction::Copy(e2_result, result));
            self.emit(Instruction::Label(end_label));
            result
        }
        .into()
    }

    fn emit_type_conversion(
        &mut self,
        src_type: &CType,
        target_type: &CType,
        src: Value,
        dest: Value,
    ) {
        if src_type == target_type {
            self.emit(Instruction::Copy(src, dest));
        } else if *target_type == CType::Double {
            if src_type.is_signed() {
                self.emit(Instruction::IntToDouble(src, dest));
            } else {
                self.emit(Instruction::UIntToDouble(src, dest));
            }
        } else if *src_type == CType::Double {
            if target_type.is_signed() {
                self.emit(Instruction::DoubleToInt(src, dest));
            } else {
                self.emit(Instruction::DoubleToUInt(src, dest));
            }
        } else if target_type.size(self.type_table) == src_type.size(self.type_table) {
            self.emit(Instruction::Copy(src, dest))
        } else if target_type.size(self.type_table) < src_type.size(self.type_table) {
            self.emit(Instruction::Truncate(src, dest))
        } else if src_type.is_signed() {
            self.emit(Instruction::SignExtend(src, dest))
        } else {
            self.emit(Instruction::ZeroExtend(src, dest))
        }
    }

    fn emit_cast(&mut self, target_type: CType, inner: Expression) -> ExpressionResult {
        let expr_type = inner.get_type().clone();

        let result = self.emit_tacky_exp_and_convert(inner);
        if target_type == expr_type {
            return result.into();
        }

        let dest = self.make_temporary(&target_type);

        self.emit_type_conversion(&expr_type, &target_type, result, dest);

        dest.into()
    }

    fn emit_tacky_exp(&mut self, code: Expression) -> ExpressionResult {
        let value_type = code.get_type().clone();

        match code.unwrap() {
            Expr::Constant(c) => c.into(),
            Expr::String(s) => self.make_string_constant(s, value_type).into(),
            Expr::Cast(CType::Void, inner) => {
                self.emit_tacky_exp_and_convert(inner);
                ExpressionResult::PlainOperand(Value::Constant(Constant::Int(0)))
            }
            Expr::Cast(target_type, inner) => self.emit_cast(target_type, inner),
            Expr::Postfix(operator, inner) => self.emit_postfix(operator, inner, value_type),
            Expr::Unary(
                operator @ (ast::UnaryOperator::PreIncrement | ast::UnaryOperator::PreDecrement),
                inner,
            ) => self.emit_pre_increment(operator, inner, value_type),
            Expr::Unary(operator, inner) => self.emit_unary(operator, inner, value_type),
            Expr::Binary(BinaryOperator::And, left, right) => self.emit_and(left, right),
            Expr::Binary(BinaryOperator::Or, left, right) => self.emit_or(left, right),
            Expr::Binary(BinaryOperator::Add, left, right) => {
                self.emit_addition(left, right, value_type)
            }
            Expr::Binary(BinaryOperator::Subtract, left, right) => {
                self.emit_subtraction(left, right, value_type)
            }
            Expr::Binary(operator, left, right) => {
                self.emit_binary(operator, left, right, value_type)
            }
            Expr::Var(name) => Value::Var(name).into(),
            Expr::Assignment(lhs, rhs) => self.emit_assignment(lhs, rhs),
            Expr::CompoundAssignment(op, lhs, rhs, name) => {
                self.emit_compound_assignment(op, lhs, rhs, name, value_type)
            }
            Expr::Conditional(condition, then_expr, else_expr) => {
                self.emit_conditional(condition, then_expr, else_expr, value_type)
            }
            Expr::FunctionCall(name, args) => {
                let arg_values = args
                    .into_iter()
                    .map(|arg| self.emit_tacky_exp_and_convert(arg))
                    .collect();
                if value_type == CType::Void {
                    self.emit(Instruction::FunCall(name, arg_values, None));

                    Value::Constant(Constant::Int(0))
                } else {
                    let result = self.make_temporary(&value_type);
                    self.emit(Instruction::FunCall(name, arg_values, Some(result)));

                    result
                }
                .into()
            }
            Expr::Dereference(inner) => {
                ExpressionResult::DereferencedPointer(self.emit_tacky_exp_and_convert(inner))
            }
            Expr::AddrOf(inner) => match self.emit_tacky_exp(inner) {
                ExpressionResult::PlainOperand(obj) => {
                    let dest = self.make_temporary(&value_type);
                    self.emit(Instruction::GetAddress(obj, dest));
                    dest.into()
                }
                ExpressionResult::DereferencedPointer(ptr) => ptr.into(),
                ExpressionResult::SubObject(base, offset) => {
                    let dest = self.make_temporary(&value_type);
                    self.emit(Instruction::GetAddress(base.into(), dest));
                    self.emit(Instruction::AddPtr(
                        dest,
                        Value::Constant(Constant::Long(offset as i64)),
                        1,
                        dest,
                    ));
                    dest.into()
                }
            },
            Expr::Subscript(left, right) => {
                let left_type = left.get_type().clone();
                let right_type = right.get_type().clone();

                let src1 = self.emit_tacky_exp_and_convert(left);
                let src2 = self.emit_tacky_exp_and_convert(right);

                if left_type.is_pointer() {
                    let dst = self.make_temporary(&left_type);
                    self.emit(Instruction::AddPtr(
                        src1,
                        src2,
                        left_type.referent_size(self.type_table),
                        dst,
                    ));
                    ExpressionResult::DereferencedPointer(dst)
                } else {
                    let dst = self.make_temporary(&right_type);
                    self.emit(Instruction::AddPtr(
                        src2,
                        src1,
                        right_type.referent_size(self.type_table),
                        dst,
                    ));
                    ExpressionResult::DereferencedPointer(dst)
                }
            }
            Expr::SizeOf(inner) => Value::Constant(Constant::ULong(
                inner.get_type().size(self.type_table) as u64,
            ))
            .into(),
            Expr::SizeOfT(t) => {
                Value::Constant(Constant::ULong(t.size(self.type_table) as u64)).into()
            }
            Expr::Dot(structure, member) => {
                let struct_type = structure.get_type();
                let struct_tag = struct_type.unwrap_structure_ref();
                let struct_def = self.type_table.get(&struct_tag.value).unwrap();
                let offset = struct_def.find_offset(member);
                match self.emit_tacky_exp(structure) {
                    ExpressionResult::PlainOperand(Value::Var(name)) => {
                        ExpressionResult::SubObject(name, offset)
                    }
                    ExpressionResult::SubObject(base, base_offset) => {
                        ExpressionResult::SubObject(base, base_offset + offset)
                    }
                    ExpressionResult::DereferencedPointer(ptr) => {
                        let dst_ptr = self.make_temporary(&CType::pointer_to(value_type.clone()));
                        self.emit(Instruction::AddPtr(
                            ptr,
                            Value::Constant(Constant::Long(offset as i64)),
                            1,
                            dst_ptr,
                        ));
                        ExpressionResult::DereferencedPointer(dst_ptr)
                    }
                    _ => panic!(),
                }
            }
            Expr::Arrow(pointer, member) => {
                let struct_tag = pointer
                    .get_type()
                    .unwrap_pointer_ref()
                    .unwrap_structure_ref();
                let offset = self
                    .type_table
                    .get(&struct_tag.value)
                    .unwrap()
                    .find_offset(member);

                let ptr_value = self.emit_tacky_exp_and_convert(pointer);

                let dest = self.make_temporary(&CType::pointer_to(value_type.clone()));

                self.emit(Instruction::AddPtr(
                    ptr_value,
                    Value::Constant(Constant::Long(offset as i64)),
                    1,
                    dest,
                ));

                ExpressionResult::DereferencedPointer(dest)
            }
        }
    }

    fn emit_tacky_exp_and_convert(&mut self, code: Expression) -> Value {
        let code_type = code.get_type().clone();
        let value = self.emit_tacky_exp(code);
        match value {
            ExpressionResult::PlainOperand(v) => v,
            ExpressionResult::DereferencedPointer(ptr) => {
                let dst = self.make_temporary(&code_type);
                self.emit(Instruction::Load(ptr, dst));
                dst
            }
            ExpressionResult::SubObject(base, offset) => {
                let dst = self.make_temporary(&code_type);
                self.emit(Instruction::CopyFromOffset(base, offset, dst));
                dst
            }
        }
    }

    fn emit_tacky_statement(&mut self, code: Statement) {
        for label in &code.labels {
            self.emit(Instruction::Label(label.clone().unwrap_plain()));
        }

        code.visit_infallible(|s| self.emit_tacky_stmt(s))
    }

    fn emit_tacky_stmt(&mut self, code: Stmt) {
        match code {
            Stmt::Expression(expr) => {
                self.emit_tacky_exp(expr);
            }
            Stmt::Return(expr) => {
                let instruction =
                    Instruction::Return(expr.map(|e| self.emit_tacky_exp_and_convert(e)));
                self.emit(instruction);
            }
            Stmt::Null => {}
            Stmt::If(condition, then_stmt, None) => {
                let end_label = self.make_label(".if_end");

                let condition_value = self.emit_tacky_exp_and_convert(condition);
                self.emit(Instruction::JumpIfZero(condition_value, end_label));
                self.emit_tacky_statement(then_stmt);
                self.emit(Instruction::Label(end_label));
            }
            Stmt::If(condition, then_stmt, Some(else_stmt)) => {
                let else_label = self.make_label(".if_else");
                let end_label = self.make_label(".if_end");

                let condition_value = self.emit_tacky_exp_and_convert(condition);
                self.emit(Instruction::JumpIfZero(condition_value, else_label));
                self.emit_tacky_statement(then_stmt);
                self.emit(Instruction::Jump(end_label));
                self.emit(Instruction::Label(else_label));
                self.emit_tacky_statement(else_stmt);
                self.emit(Instruction::Label(end_label));
            }
            Stmt::Compound(block) => {
                self.emit_tacky_block(block);
            }
            Stmt::Break(label) => {
                self.emit(Instruction::Jump(convert_loop_label(
                    BREAK_TAG,
                    label.expect("Unlabelled break"),
                )));
            }

            Stmt::Continue(label) => {
                self.emit(Instruction::Jump(convert_loop_label(
                    CONTINUE_TAG,
                    label.expect("Unlabelled continue"),
                )));
            }

            Stmt::DoWhile(body, condition, label) => {
                let label = label.expect("Unlabelled do while");

                let start_label = self.make_label(".do_start");

                self.emit(Instruction::Label(start_label));
                self.emit_tacky_statement(body);
                self.emit(Instruction::Label(convert_loop_label(CONTINUE_TAG, label)));
                let result = self.emit_tacky_exp_and_convert(condition);
                self.emit(Instruction::JumpIfNotZero(result, start_label));
                self.emit(Instruction::Label(convert_loop_label(BREAK_TAG, label)));
            }

            Stmt::While(condition, body, label) => {
                let label = label.expect("Unlabelled while");

                let continue_label = convert_loop_label(CONTINUE_TAG, label);
                let break_label = convert_loop_label(BREAK_TAG, label);

                self.emit(Instruction::Label(continue_label));
                let result = self.emit_tacky_exp_and_convert(condition);
                self.emit(Instruction::JumpIfZero(result, break_label));
                self.emit_tacky_statement(body);
                self.emit(Instruction::Jump(continue_label));
                self.emit(Instruction::Label(break_label));
            }

            Stmt::For(init, cond, incr, body, label) => {
                let label = label.expect("Unlabelled for");

                let start_label = self.make_label(".for_start");
                let continue_label = convert_loop_label(CONTINUE_TAG, label);
                let break_label = convert_loop_label(BREAK_TAG, label);

                match *init {
                    ForInit::InitDecl(decl) => self.emit_tacky_var_decl(decl),
                    ForInit::InitExp(Some(expr)) => {
                        self.emit_tacky_exp(expr);
                    }
                    ForInit::InitExp(None) => {}
                }

                self.emit(Instruction::Label(start_label));

                if let Some(condition) = cond {
                    let cond_result = self.emit_tacky_exp_and_convert(condition);
                    self.emit(Instruction::JumpIfZero(cond_result, break_label));
                }

                self.emit_tacky_statement(body);
                self.emit(Instruction::Label(continue_label));

                if let Some(increment) = incr {
                    self.emit_tacky_exp(increment);
                }
                self.emit(Instruction::Jump(start_label));
                self.emit(Instruction::Label(break_label));
            }
            Stmt::Goto(label) => self.emit(Instruction::Jump(label)),
            Stmt::Switch(expr, body, cases, default_label, label) => {
                let expr_type = expr.get_type().clone();
                let break_label = convert_loop_label(BREAK_TAG, label.unwrap());

                let switch_value = self.emit_tacky_exp_and_convert(expr);
                for case in cases {
                    let compare_result = self.make_temporary(&expr_type);
                    self.emit(Instruction::Binary(
                        BinaryOp::Equal,
                        switch_value,
                        case.value.into(),
                        compare_result,
                    ));
                    self.emit(Instruction::JumpIfNotZero(compare_result, case.label));
                }

                self.emit(Instruction::Jump(default_label.unwrap_or(break_label)));

                self.emit_tacky_statement(body);
                self.emit(Instruction::Label(break_label));
            }
        }
    }

    fn emit_tacky_compound_init(&mut self, name: Identifier, init: Initializer, mut offset: usize) {
        let init_type = init.get_type().clone();
        match (init, init_type) {
            (init, CType::Array(..)) if init.is_string() => {
                let size = init.get_type().size(self.type_table);
                let s = init.get_string();
                for constant in chunk_string_init(&s, size) {
                    let size = constant.size();
                    self.emit(Instruction::CopyToOffset(constant.into(), name, offset));
                    offset += size;
                }
            }
            (init, _) if init.is_single() => {
                let val = self.emit_tacky_exp_and_convert(init.unwrap_single());
                self.emit(Instruction::CopyToOffset(val, name, offset));
            }
            (init, CType::Structure(tag)) if init.is_compound() => {
                let init_list = init.unwrap_compound();
                let members = &self.type_table.get(&tag.value).unwrap().members;
                for (member_init, member) in init_list.into_iter().zip(members.iter()) {
                    self.emit_tacky_compound_init(name, member_init, offset + member.offset);
                }
            }
            (init, CType::Array(element, _)) if init.is_compound() => {
                let size = element.size(self.type_table);
                let init_list = init.unwrap_compound();
                for initializer in init_list {
                    self.emit_tacky_compound_init(name, initializer, offset);
                    offset += size;
                }
            }
            l => {
                eprintln!("{:?}", l);
                panic!()
            }
        }
    }

    fn emit_tacky_var_decl(&mut self, code: VarDeclaration) {
        if code.storage_class != StorageClass::None {
            return;
        }
        if let Some(init) = code.init {
            if init.is_single() {
                let val = self.emit_tacky_exp_and_convert(init.unwrap_single());
                self.emit(Instruction::Copy(val, code.name.into()));
            } else {
                self.emit_tacky_compound_init(code.name, init, 0);
            }
        }
    }

    fn emit_tacky_block(&mut self, code: Block) {
        for block_item in code.0 {
            match block_item {
                BlockItem::S(stmt) => self.emit_tacky_statement(stmt),
                BlockItem::D(Declaration::Var(decl)) => self.emit_tacky_var_decl(decl),
                BlockItem::D(Declaration::Fn(function)) => {
                    if function.body.is_some() {
                        panic!("Unexpected function body");
                    }
                }
                BlockItem::D(Declaration::Struct(..)) => {}
            }
        }
    }

    fn emit_tacky_function(&mut self, code: FunctionDeclaration) -> Option<ir::Function> {
        code.body.map(|block| {
            self.emit_tacky_block(block);

            if code.fun_type.ret == CType::Void {
                self.emit(Instruction::Return(None))
            } else {
                self.emit(Instruction::Return(Some(Constant::ZERO.into())));
            }

            let symbol_entry = self
                .symbol_table
                .get(&code.name.value)
                .expect("Function should be in symbol table");

            ir::Function {
                name: code.name,
                global: symbol_entry.attrs.unwrap_fun_ref().global,
                params: code.params.into_iter().map(Into::into).collect(),
                body: mem::take(&mut self.instructions),
            }
        })
    }

    fn emit_tacky_declaration(&mut self, code: Declaration) -> Option<ir::Function> {
        match code {
            Declaration::Fn(f) => self.emit_tacky_function(f),
            Declaration::Var(_) => None,
            Declaration::Struct(_) => None,
        }
    }

    fn emit_symbol_tacky(&self) -> Vec<ir::TopLevel> {
        let mut tacky_defs = vec![];

        for (name, entry) in &*self.symbol_table {
            if let IdentifierAttrs::Static(StaticAttr { init, global }) = &entry.attrs {
                match init {
                    InitialValue::Initial(i) => tacky_defs.push(
                        ir::StaticVariable {
                            name: (*name).into(),
                            global: *global,
                            var_type: entry.c_type.clone(),
                            init_list: i.clone(),
                        }
                        .into(),
                    ),
                    InitialValue::Tentative => tacky_defs.push(
                        ir::StaticVariable {
                            name: (*name).into(),
                            global: *global,
                            init_list: vec![StaticInit::zero(&entry.c_type, self.type_table)],
                            var_type: entry.c_type.clone(),
                        }
                        .into(),
                    ),
                    InitialValue::NoInitializer => {}
                }
            } else if let IdentifierAttrs::Const(i) = &entry.attrs {
                tacky_defs.push(
                    ir::StaticConstant {
                        name: (*name).into(),
                        var_type: entry.c_type.clone(),
                        init: i.clone(),
                    }
                    .into(),
                )
            };
        }

        tacky_defs
    }
}

pub fn tackify_program(
    code: Program,
    symbol_table: &mut SymbolTable,
    type_table: &TypeTable,
) -> ir::Program {
    let mut tackier = Tackier::new(symbol_table, type_table);

    let mut top_level: Vec<ir::TopLevel> = code
        .declarations
        .into_iter()
        .filter_map(|decl| tackier.emit_tacky_declaration(decl))
        .map(|function| function.into())
        .collect();

    top_level.extend_from_slice(&tackier.emit_symbol_tacky());

    ir::Program { top_level }
}

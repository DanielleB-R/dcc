use std::cmp::max;
use std::collections::HashSet;

use crate::common::Identifier;
use crate::common::ctype::*;
use crate::common::symbol_table::*;
use crate::common::type_table::*;
use crate::errors::TypecheckError;
use crate::parser::ast::*;
use BinaryOperator::*;

pub type Result<T> = std::result::Result<T, TypecheckError>;

fn get_common_pointer_type(e1: &Expression, e2: &Expression, line: usize) -> Result<CType> {
    let e1_type = e1.get_type();
    let e2_type = e2.get_type();

    if e1_type == e2_type {
        Ok(e1_type.clone())
    } else if e1.is_null_pointer_constant() {
        Ok(e2_type.clone())
    } else if e2.is_null_pointer_constant() {
        Ok(e1_type.clone())
    } else if (e1_type.is_void_pointer() && e2_type.is_pointer())
        || (e1_type.is_pointer() && e2_type.is_void_pointer())
    {
        Ok(CType::pointer_to(CType::Void))
    } else {
        Err(TypecheckError::IncompatibleTypes(
            e1_type.clone(),
            e2_type.clone(),
            line,
        ))
    }
}

fn convert_to(expr: Expression, target_type: &CType) -> Expression {
    if expr.get_type() == target_type {
        expr
    } else {
        let mut cast_exp: Expression = Expr::Cast(target_type.clone(), expr).into();
        cast_exp.set_type(target_type.clone());
        cast_exp
    }
}

fn should_convert_to_target(expr: &Expression, target_type: &CType) -> bool {
    (expr.get_type().is_arithmetic() && target_type.is_arithmetic())
        || (expr.is_null_pointer_constant() && target_type.is_pointer())
        || (expr.get_type().is_void_pointer() && target_type.is_pointer())
        || (expr.get_type().is_pointer() && target_type.is_void_pointer())
}

fn convert_by_assignment(expr: Expression, target_type: &CType, line: usize) -> Result<Expression> {
    if expr.get_type() == target_type {
        Ok(expr)
    } else if should_convert_to_target(&expr, target_type) {
        Ok(convert_to(expr, target_type))
    } else {
        Err(TypecheckError::IncompatibleTypes(
            target_type.clone(),
            expr.get_type().clone(),
            line,
        ))
    }
}

fn convert_string_initializer(
    s: String,
    array_type: &CType,
    line: usize,
) -> Result<Vec<StaticInit>> {
    let (element_type, size) = array_type.unwrap_array_ref();
    if !element_type.is_character() {
        return Err(TypecheckError::CannotInitializeNonCharacter(line));
    }

    let string_length = s.len();
    if string_length > *size {
        Err(TypecheckError::WrongInitializerLength(line))
    } else if string_length == *size {
        Ok(vec![StaticInit::StringInit(s, false)])
    } else if string_length == *size - 1 {
        Ok(vec![StaticInit::StringInit(s, true)])
    } else {
        Ok(vec![
            StaticInit::StringInit(s, true),
            StaticInit::ZeroInit(*size - string_length - 1),
        ])
    }
}

struct Typechecker {
    symbols: SymbolTable,
    types: TypeTable,
    current_return_type: Option<CType>,
    constant_index: usize,
}

impl Typechecker {
    fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            types: TypeTable::new(),
            current_return_type: None,
            constant_index: 0,
        }
    }

    fn generate_constant_name(&mut self) -> String {
        self.constant_index += 1;
        format!(".string.{}", self.constant_index)
    }

    fn get_common_type(&self, mut type1: &CType, mut type2: &CType) -> CType {
        if type1.is_character() {
            type1 = &CType::Int;
        }
        if type2.is_character() {
            type2 = &CType::Int;
        }

        if type1 == type2 {
            type1.clone()
        } else if *type1 == CType::Double || *type2 == CType::Double {
            CType::Double
        } else if type1.size(&self.types) == type2.size(&self.types) {
            if type1.is_signed() {
                type2.clone()
            } else {
                type1.clone()
            }
        } else if type1.size(&self.types) > type2.size(&self.types) {
            type1.clone()
        } else {
            type2.clone()
        }
    }

    fn convert_static_compound_initializers(
        &mut self,
        initializers: &[Initializer],
        var_type: &CType,
    ) -> Result<Vec<StaticInit>> {
        let (value_type, size) = var_type.unwrap_array_ref();

        if initializers.len() > *size {
            return Err(TypecheckError::WrongInitializerLength(
                initializers[0].get_line(),
            ));
        }

        let mut result = vec![];

        if value_type.is_array() {
            for initializer in initializers {
                if initializer.is_string() {
                    result.extend(convert_string_initializer(
                        initializer.clone().get_string(),
                        value_type,
                        initializer.get_line(),
                    )?);
                } else if initializer.is_single() {
                    return Err(TypecheckError::WrongInitializerLength(
                        initializer.get_line(),
                    ));
                } else {
                    result.extend(self.convert_static_compound_initializers(
                        initializer.unwrap_compound_ref(),
                        value_type,
                    )?);
                }
            }

            let remaining_slots = *size - initializers.len();
            if remaining_slots != 0 {
                result.push(StaticInit::ZeroInit(
                    remaining_slots * value_type.size(&self.types),
                ));
            }
        } else if value_type.is_structure() {
            for init_elem in initializers {
                result.extend(
                    self.typecheck_constant_initializer(init_elem, value_type)?
                        .unwrap_initial()
                        .into_iter(),
                );
            }

            let remaining_slots = *size - initializers.len();
            if remaining_slots != 0 {
                result.push(StaticInit::ZeroInit(
                    remaining_slots * value_type.size(&self.types),
                ));
            }
        } else {
            for initializer in initializers {
                if initializer.is_compound() {
                    return Err(TypecheckError::WrongInitializerLength(
                        initializer.get_line(),
                    ));
                }
                if value_type.is_pointer()
                    && !initializer.unwrap_single_ref().is_null_pointer_constant()
                {
                    return Err(TypecheckError::MiscError(
                        "Cannot initialize static pointer with non-null number",
                        initializer.get_line(),
                    ));
                }
                result.push(StaticInit::from_constant(
                    initializer.clone().get_constant(),
                    value_type,
                ));
            }
            let remaining_slots = *size - initializers.len();
            if remaining_slots != 0 {
                result.push(StaticInit::ZeroInit(
                    remaining_slots * value_type.size(&self.types),
                ))
            }
        }

        Ok(result)
    }

    fn validate_value_type_specifier(&self, t: &CType) -> Result<()> {
        match t {
            CType::Array(element_type, _) => {
                if element_type.is_complete(&self.types) {
                    self.validate_value_type_specifier(element_type)
                } else {
                    Err(TypecheckError::ArrayOfIncomplete)
                }
            }
            CType::Pointer(referenced_type) => self.validate_value_type_specifier(referenced_type),
            CType::Function(f) => self.validate_function_type_specifier(f),
            _ => Ok(()),
        }
    }

    fn validate_function_type_specifier(&self, t: &FunctionType) -> Result<()> {
        for param in &t.params {
            self.validate_value_type_specifier(param)?;
        }
        self.validate_value_type_specifier(&t.ret)
    }

    fn typecheck_function_call(
        &mut self,
        name: Identifier,
        args: Vec<Expression>,
        line: usize,
    ) -> Result<(Expr, CType)> {
        let function_type = self.symbols.get_expected_type(name.value);

        if !function_type.is_function() {
            return Err(TypecheckError::CalledNonCallable(line));
        }
        let function_type = function_type.unwrap_function();

        if function_type.params.len() != args.len() {
            return Err(TypecheckError::NonMatchingArguments(line));
        }

        if function_type.ret != CType::Void && !function_type.ret.is_complete(&self.types) {
            return Err(TypecheckError::MiscError(
                "Cannot call a function returning an incomplete type",
                line,
            ));
        }

        let converted_args = args
            .into_iter()
            .map(|arg| self.typecheck_and_convert(arg))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .zip(function_type.params)
            .map(|(arg, param)| convert_by_assignment(arg, &param, line))
            .collect::<Result<Vec<_>>>()?;

        Ok((Expr::FunctionCall(name, converted_args), function_type.ret))
    }

    fn typecheck_addition(&mut self, left: Expression, right: Expression) -> Result<(Expr, CType)> {
        let typed_left = self.typecheck_and_convert(left)?;
        let typed_right = self.typecheck_and_convert(right)?;
        let left_type = typed_left.get_type();
        let right_type = typed_right.get_type();

        if left_type.is_arithmetic() && right_type.is_arithmetic() {
            let common_type = self.get_common_type(left_type, right_type);
            Ok((
                Expr::Binary(
                    Add,
                    convert_to(typed_left, &common_type),
                    convert_to(typed_right, &common_type),
                ),
                common_type,
            ))
        } else if left_type.is_pointer_to_complete(&self.types) && right_type.is_integer() {
            let left_type = left_type.clone();
            Ok((
                Expr::Binary(Add, typed_left, convert_to(typed_right, &CType::Long)),
                left_type,
            ))
        } else if left_type.is_integer() && right_type.is_pointer_to_complete(&self.types) {
            let right_type = right_type.clone();
            Ok((
                Expr::Binary(Add, convert_to(typed_left, &CType::Long), typed_right),
                right_type,
            ))
        } else {
            Err(TypecheckError::BadPointerArithmetic(typed_left.get_line()))
        }
    }

    fn typecheck_subtraction(
        &mut self,
        left: Expression,
        right: Expression,
    ) -> Result<(Expr, CType)> {
        let typed_left = self.typecheck_and_convert(left)?;
        let typed_right = self.typecheck_and_convert(right)?;
        let left_type = typed_left.get_type();
        let right_type = typed_right.get_type();

        if left_type.is_arithmetic() && right_type.is_arithmetic() {
            let common_type = self.get_common_type(left_type, right_type);
            Ok((
                Expr::Binary(
                    Subtract,
                    convert_to(typed_left, &common_type),
                    convert_to(typed_right, &common_type),
                ),
                common_type,
            ))
        } else if left_type.is_pointer_to_complete(&self.types) && right_type.is_integer() {
            let left_type = left_type.clone();
            Ok((
                Expr::Binary(Subtract, typed_left, convert_to(typed_right, &CType::Long)),
                left_type,
            ))
        } else if left_type.is_pointer_to_complete(&self.types) && left_type == right_type {
            Ok((Expr::Binary(Subtract, typed_left, typed_right), CType::Long))
        } else {
            Err(TypecheckError::BadPointerArithmetic(typed_left.get_line()))
        }
    }

    fn typecheck_conditional(
        &mut self,
        left: Expression,
        middle: Expression,
        right: Expression,
        line: usize,
    ) -> Result<(Expr, CType)> {
        let typed_left = self.typecheck_and_convert(left)?;
        let typed_middle = self.typecheck_and_convert(middle)?;
        let typed_right = self.typecheck_and_convert(right)?;

        if !typed_left.get_type().is_scalar() {
            return Err(TypecheckError::ConditionMustBeScalar);
        }

        let middle_type = typed_middle.get_type();
        let right_type = typed_right.get_type();

        let common_type = if *middle_type == CType::Void && *right_type == CType::Void {
            CType::Void
        } else if middle_type.is_arithmetic() && right_type.is_arithmetic() {
            self.get_common_type(middle_type, right_type)
        } else if middle_type.is_pointer() || right_type.is_pointer() {
            get_common_pointer_type(&typed_middle, &typed_right, line)?
        } else if middle_type.is_structure()
            && right_type.is_structure()
            && middle_type == right_type
        {
            middle_type.clone()
        } else {
            return Err(TypecheckError::CannotConvertBranches(line));
        };

        Ok((
            Expr::Conditional(
                typed_left,
                convert_to(typed_middle, &common_type),
                convert_to(typed_right, &common_type),
            ),
            common_type,
        ))
    }

    fn typecheck_exp(&mut self, expr: Expression) -> Result<Expression> {
        expr.type_map(|e, line| self.typecheck_expr(e, line))
    }

    fn typecheck_expr(&mut self, expr: Expr, line: usize) -> Result<(Expr, CType)> {
        Ok(match expr {
            Expr::FunctionCall(function, args) => {
                self.typecheck_function_call(function, args, line)?
            }
            Expr::Var(name) => {
                let symbol_type = self.symbols.get_expected_type(name.value);

                if symbol_type.is_function() {
                    return Err(TypecheckError::ValueOfCallable(line));
                }
                (Expr::Var(name), symbol_type)
            }
            Expr::Constant(e) => (Expr::Constant(e), e.get_type()),
            Expr::String(s) => {
                let string_type = CType::for_string(&s);
                (Expr::String(s), string_type)
            }
            Expr::Cast(t, inner) => {
                self.validate_value_type_specifier(&t)?;
                let typed_inner = self.typecheck_and_convert(inner)?;

                if (t == CType::Double && typed_inner.get_type().is_pointer())
                    || (t.is_pointer() && *typed_inner.get_type() == CType::Double)
                {
                    return Err(TypecheckError::PointerDoubleCast(line));
                }
                if t == CType::Void {
                    (Expr::Cast(t, typed_inner), CType::Void)
                } else if !t.is_scalar() {
                    return Err(TypecheckError::NonScalarCast(line));
                } else if !typed_inner.get_type().is_scalar() {
                    return Err(TypecheckError::CastNonScalar(line));
                } else {
                    (Expr::Cast(t.clone(), typed_inner), t)
                }
            }
            Expr::Unary(
                op @ (UnaryOperator::PreIncrement | UnaryOperator::PreDecrement),
                operand,
            ) => {
                let typed_inner = self.typecheck_and_convert(operand)?;

                if !typed_inner.is_lvalue() {
                    return Err(TypecheckError::NonLvalueAssignment(line));
                }
                let value_type = typed_inner.get_type().clone();

                if value_type.is_arithmetic() || value_type.is_pointer_to_complete(&self.types) {
                    (Expr::Unary(op, typed_inner), value_type)
                } else {
                    return Err(TypecheckError::BadIncrement(line));
                }
            }

            Expr::Unary(UnaryOperator::Not, operand) => {
                let typed_inner = self.typecheck_and_convert(operand)?;

                if typed_inner.get_type().is_scalar() {
                    (Expr::Unary(UnaryOperator::Not, typed_inner), CType::Int)
                } else {
                    return Err(TypecheckError::LogicalRequiresScalar(
                        typed_inner.get_line(),
                    ));
                }
            }

            // Doesn't match increment/decrement
            Expr::Unary(op, operand) => {
                let typed_inner = self.typecheck_and_convert(operand)?;
                let value_type = typed_inner.get_type().clone();

                if value_type.is_double() && op == UnaryOperator::Complement {
                    return Err(TypecheckError::BitwiseOpOnDouble(line));
                }

                if value_type.is_arithmetic() {
                    if value_type.is_character() {
                        (
                            Expr::Unary(op, convert_to(typed_inner, &CType::Int)),
                            CType::Int,
                        )
                    } else {
                        (Expr::Unary(op, typed_inner), value_type)
                    }
                } else {
                    return Err(TypecheckError::BadPointerArithmetic(line));
                }
            }
            Expr::Binary(op @ (Equal | NotEqual), left, right) => {
                let typed_left = self.typecheck_and_convert(left)?;
                let typed_right = self.typecheck_and_convert(right)?;
                let common_type =
                    if typed_left.get_type().is_pointer() || typed_right.get_type().is_pointer() {
                        get_common_pointer_type(&typed_left, &typed_right, line)?
                    } else if typed_left.get_type().is_arithmetic()
                        && typed_right.get_type().is_arithmetic()
                    {
                        self.get_common_type(typed_left.get_type(), typed_right.get_type())
                    } else {
                        return Err(TypecheckError::MiscError(
                            "Invalid operands to equality expression",
                            line,
                        ));
                    };

                (
                    Expr::Binary(
                        op,
                        convert_to(typed_left, &common_type),
                        convert_to(typed_right, &common_type),
                    ),
                    CType::Int,
                )
            }

            Expr::Binary(op @ (And | Or), left, right) => {
                let typed_left = self.typecheck_and_convert(left)?;
                let typed_right = self.typecheck_and_convert(right)?;
                if typed_left.get_type().is_scalar() && typed_right.get_type().is_scalar() {
                    (Expr::Binary(op, typed_left, typed_right), CType::Int)
                } else {
                    return Err(TypecheckError::LogicalRequiresScalar(typed_left.get_line()));
                }
            }

            Expr::Binary(op, left, right) if op.is_shift() => {
                let mut typed_left = self.typecheck_and_convert(left)?;
                let typed_right = self.typecheck_and_convert(right)?;
                if typed_left.get_type().is_double() || typed_right.get_type().is_double() {
                    return Err(TypecheckError::BitwiseOpOnDouble(line));
                }
                if typed_left.get_type().is_arithmetic() && typed_right.get_type().is_arithmetic() {
                    if typed_left.get_type().is_character() {
                        typed_left = convert_to(typed_left, &CType::Int);
                    }

                    let expr_type = typed_left.get_type().clone();
                    (Expr::Binary(op, typed_left, typed_right), expr_type)
                } else {
                    return Err(TypecheckError::BadPointerArithmetic(line));
                }
            }

            Expr::Binary(Add, left, right) => self.typecheck_addition(left, right)?,

            Expr::Binary(Subtract, left, right) => self.typecheck_subtraction(left, right)?,

            Expr::Binary(op @ (Multiply | Divide | Remainder), left, right) => {
                let typed_left = self.typecheck_and_convert(left)?;
                let typed_right = self.typecheck_and_convert(right)?;
                if typed_left.get_type().is_arithmetic() && typed_right.get_type().is_arithmetic() {
                    let common_type =
                        self.get_common_type(typed_left.get_type(), typed_right.get_type());

                    if common_type == CType::Double && op == Remainder {
                        return Err(TypecheckError::BitwiseOpOnDouble(line));
                    }

                    (
                        Expr::Binary(
                            op,
                            convert_to(typed_left, &common_type),
                            convert_to(typed_right, &common_type),
                        ),
                        common_type,
                    )
                } else {
                    return Err(TypecheckError::BadPointerArithmetic(line));
                }
            }

            // This doesn't include shifts because they're matched above
            Expr::Binary(op, left, right) if op.is_bitwise() => {
                let typed_left = self.typecheck_and_convert(left)?;
                let typed_right = self.typecheck_and_convert(right)?;
                if typed_left.get_type().is_arithmetic() && typed_right.get_type().is_arithmetic() {
                    let common_type =
                        self.get_common_type(typed_left.get_type(), typed_right.get_type());

                    if common_type == CType::Double {
                        return Err(TypecheckError::BitwiseOpOnDouble(line));
                    }

                    (
                        Expr::Binary(
                            op,
                            convert_to(typed_left, &common_type),
                            convert_to(typed_right, &common_type),
                        ),
                        common_type,
                    )
                } else {
                    return Err(TypecheckError::BadPointerArithmetic(line));
                }
            }

            // This only leaves the relational operators
            Expr::Binary(op, left, right) => {
                let typed_left = self.typecheck_and_convert(left)?;
                let typed_right = self.typecheck_and_convert(right)?;

                if typed_left.get_type().is_arithmetic() && typed_right.get_type().is_arithmetic() {
                    let common_type =
                        self.get_common_type(typed_left.get_type(), typed_right.get_type());

                    (
                        Expr::Binary(
                            op,
                            convert_to(typed_left, &common_type),
                            convert_to(typed_right, &common_type),
                        ),
                        CType::Int,
                    )
                } else if typed_left.get_type().is_pointer()
                    && typed_left.get_type() == typed_right.get_type()
                {
                    (Expr::Binary(op, typed_left, typed_right), CType::Int)
                } else {
                    return Err(TypecheckError::BadComparison(typed_left.get_line()));
                }
            }
            Expr::Postfix(op, operand) => {
                let typed_inner = self.typecheck_and_convert(operand)?;
                if !typed_inner.is_lvalue() {
                    return Err(TypecheckError::NonLvalueAssignment(line));
                }

                let inner_type = typed_inner.get_type().clone();

                if inner_type.is_arithmetic() || inner_type.is_pointer_to_complete(&self.types) {
                    (Expr::Postfix(op, typed_inner), inner_type)
                } else {
                    return Err(TypecheckError::BadIncrement(line));
                }
            }
            Expr::Assignment(target, value) => {
                let typed_target = self.typecheck_and_convert(target)?;
                if !typed_target.is_lvalue() {
                    return Err(TypecheckError::NonLvalueAssignment(line));
                }
                let typed_value = self.typecheck_and_convert(value)?;

                let target_type = typed_target.get_type().clone();
                (
                    Expr::Assignment(
                        typed_target,
                        convert_by_assignment(typed_value, &target_type, line)?,
                    ),
                    target_type,
                )
            }
            Expr::CompoundAssignment(op, target, value, _) => {
                if !target.is_lvalue() {
                    return Err(TypecheckError::NonLvalueAssignment(line));
                }

                let typed_target = self.typecheck_and_convert(target)?;
                if !typed_target.is_lvalue() {
                    return Err(TypecheckError::NonLvalueAssignment(line));
                }
                let typed_value = self.typecheck_and_convert(value)?;

                let expr_type = typed_target.get_type().clone();

                if typed_value.get_type().is_pointer() {
                    return Err(TypecheckError::BadPointerArithmetic(line));
                }
                if typed_value.get_type().is_void() {
                    return Err(TypecheckError::MiscError(
                        "Cannot have void on the right hand side of compound assignment",
                        line,
                    ));
                }
                if typed_value.get_type().is_structure() {
                    return Err(TypecheckError::MiscError(
                        "Cannot have structure on the right hand side of compound assignment",
                        line,
                    ));
                }

                if expr_type.is_pointer() {
                    if typed_value.get_type().is_double()
                        || (op == Multiply || op == Divide || op == Remainder || op.is_bitwise())
                        || !expr_type.is_pointer_to_complete(&self.types)
                    {
                        return Err(TypecheckError::BadPointerArithmetic(line));
                    } else {
                        return Ok((
                            Expr::CompoundAssignment(op, typed_target, typed_value, None),
                            expr_type,
                        ));
                    }
                }

                if op.is_shift() {
                    if *typed_target.get_type() == CType::Double
                        || *typed_value.get_type() == CType::Double
                    {
                        return Err(TypecheckError::BitwiseOpOnDouble(line));
                    }
                    return Ok((
                        Expr::CompoundAssignment(op, typed_target, typed_value, None),
                        expr_type,
                    ));
                }

                let target_type = self.get_common_type(&expr_type, typed_value.get_type());

                if target_type == CType::Double && (op.is_bitwise() || op == Remainder) {
                    return Err(TypecheckError::BitwiseOpOnDouble(line));
                }

                let converted_value = convert_by_assignment(typed_value, &target_type, line)?;

                (
                    Expr::CompoundAssignment(
                        op,
                        typed_target,
                        converted_value,
                        if expr_type != target_type {
                            Some(target_type)
                        } else {
                            None
                        },
                    ),
                    expr_type.clone(),
                )
            }
            Expr::Conditional(left, middle, right) => {
                self.typecheck_conditional(left, middle, right, line)?
            }
            Expr::Dereference(inner) => {
                let typed_inner = self.typecheck_and_convert(inner)?;
                match typed_inner.get_type().clone() {
                    CType::Pointer(referenced) => {
                        if *referenced == CType::Void {
                            return Err(TypecheckError::NoDereferenceToVoid);
                        } else {
                            (Expr::Dereference(typed_inner), *referenced)
                        }
                    }
                    _ => return Err(TypecheckError::NonPointerDereference(line)),
                }
            }
            Expr::AddrOf(inner) => {
                if inner.is_lvalue() {
                    let typed_inner = self.typecheck_exp(inner)?;
                    let pointer_type = CType::pointer_to(typed_inner.get_type().clone());
                    (Expr::AddrOf(typed_inner), pointer_type)
                } else {
                    return Err(TypecheckError::NonLvalueAddress(line));
                }
            }
            Expr::Subscript(left, right) => {
                let mut typed_left = self.typecheck_and_convert(left)?;
                let mut typed_right = self.typecheck_and_convert(right)?;
                let left_type = typed_left.get_type();
                let right_type = typed_right.get_type();
                let ptr_type = if left_type.is_pointer_to_complete(&self.types)
                    && right_type.is_integer()
                {
                    typed_right = convert_to(typed_right, &CType::Long);
                    left_type
                } else if left_type.is_integer() && right_type.is_pointer_to_complete(&self.types) {
                    typed_left = convert_to(typed_left, &CType::Long);
                    right_type
                } else {
                    return Err(TypecheckError::BadSubscript(typed_left.get_line()));
                }
                .clone();

                (
                    Expr::Subscript(typed_left, typed_right),
                    *ptr_type.unwrap_pointer(),
                )
            }
            Expr::SizeOf(inner) => {
                let typed_inner = self.typecheck_exp(inner)?;

                if !typed_inner.get_type().is_complete(&self.types) {
                    return Err(TypecheckError::MiscError(
                        "Can't get the size of an incomplete type",
                        line,
                    ));
                }
                (Expr::SizeOf(typed_inner), CType::UnsignedLong)
            }
            Expr::SizeOfT(t) => {
                self.validate_value_type_specifier(&t)?;
                if !t.is_complete(&self.types) {
                    return Err(TypecheckError::MiscError(
                        "Can't get the size of an incomplete type",
                        line,
                    ));
                }

                (Expr::SizeOfT(t), CType::UnsignedLong)
            }
            Expr::Dot(expr, member) => {
                let typed_structure = self.typecheck_and_convert(expr)?;
                let structure_type = typed_structure.get_type().clone();
                match structure_type {
                    CType::Structure(tag) => {
                        let definition = self.types.get(&tag.value).unwrap();
                        let member_def = definition
                            .members
                            .iter()
                            .find(|m| m.name.value == member.value);
                        match member_def {
                            Some(member) => (
                                Expr::Dot(typed_structure, member.name),
                                member.member_type.clone(),
                            ),
                            None => {
                                return Err(TypecheckError::MiscError(
                                    "Structure has no member with this name",
                                    typed_structure.get_line(),
                                ));
                            }
                        }
                    }
                    _ => {
                        return Err(TypecheckError::MiscError(
                            "Tried to get member of non-structure",
                            typed_structure.get_line(),
                        ));
                    }
                }
            }
            Expr::Arrow(expr, member) => {
                let typed_ptr = self.typecheck_and_convert(expr)?;
                let ptr_type = typed_ptr.get_type().clone();

                if !ptr_type.is_pointer() || !ptr_type.unwrap_pointer_ref().is_structure() {
                    return Err(TypecheckError::MiscError(
                        "Tried to get member of non-structure",
                        typed_ptr.get_line(),
                    ));
                }

                let tag = ptr_type.unwrap_pointer().unwrap_structure();

                let definition = self.types.get(&tag.value).unwrap();
                let member_def = definition
                    .members
                    .iter()
                    .find(|m| m.name.value == member.value);
                match member_def {
                    Some(member) => (
                        Expr::Arrow(typed_ptr, member.name),
                        member.member_type.clone(),
                    ),
                    None => {
                        return Err(TypecheckError::MiscError(
                            "Structure has no member with this name",
                            typed_ptr.get_line(),
                        ));
                    }
                }
            }
        })
    }

    fn typecheck_and_convert(&mut self, expr: Expression) -> Result<Expression> {
        let typed_expr = self.typecheck_exp(expr)?;
        match typed_expr.get_type() {
            CType::Array(element_type, _) => {
                let element_type = element_type.clone();
                let mut addr_expr: Expression = Expr::AddrOf(typed_expr).into();
                addr_expr.set_type(CType::Pointer(element_type));
                Ok(addr_expr)
            }
            CType::Structure(tag) => {
                if let Some(struct_type) = self.types.get(&tag.value)
                    && !struct_type.members.is_empty()
                {
                    Ok(typed_expr)
                } else {
                    Err(TypecheckError::MiscError(
                        "Invalid use of incomplete structure type",
                        typed_expr.get_line(),
                    ))
                }
            }
            _ => Ok(typed_expr),
        }
    }

    fn typecheck_statement(&mut self, statement: Statement) -> Result<Statement> {
        statement.map(|s| self.typecheck_stmt(s))
    }

    fn typecheck_stmt(&mut self, stmt: Stmt) -> Result<Stmt> {
        Ok(match stmt {
            Stmt::Return(expr) => {
                let is_void = *self.current_return_type.as_ref().unwrap() == CType::Void;
                let has_return = expr.is_some();
                if is_void && has_return {
                    return Err(TypecheckError::CannotReturnFromVoid);
                }

                if !is_void && !has_return {
                    return Err(TypecheckError::MustReturnFromNonVoid);
                }

                Stmt::Return(
                    expr.map(|e| {
                        convert_by_assignment(
                            self.typecheck_and_convert(e)?,
                            self.current_return_type.as_ref().unwrap(),
                            0,
                        )
                    })
                    .transpose()?,
                )
            }

            Stmt::Expression(expr) => Stmt::Expression(self.typecheck_and_convert(expr)?),
            Stmt::If(cond, then_stmt, else_stmt) => {
                let typed_cond = self.typecheck_exp(cond)?;

                if !typed_cond.get_type().is_scalar() {
                    return Err(TypecheckError::ConditionMustBeScalar);
                }

                Stmt::If(
                    typed_cond,
                    self.typecheck_statement(then_stmt)?,
                    else_stmt.map(|e| self.typecheck_statement(e)).transpose()?,
                )
            }
            Stmt::Compound(block) => Stmt::Compound(self.typecheck_block(block)?),
            Stmt::While(cond, body, label) => {
                let typed_cond = self.typecheck_exp(cond)?;

                if !typed_cond.get_type().is_scalar() {
                    return Err(TypecheckError::ConditionMustBeScalar);
                }

                Stmt::While(typed_cond, self.typecheck_statement(body)?, label)
            }
            Stmt::DoWhile(body, cond, label) => {
                let typed_cond = self.typecheck_exp(cond)?;

                if !typed_cond.get_type().is_scalar() {
                    return Err(TypecheckError::ConditionMustBeScalar);
                }

                Stmt::DoWhile(self.typecheck_statement(body)?, typed_cond, label)
            }
            Stmt::For(init, cond, incr, body, label) => {
                let init = Box::new(match *init {
                    ForInit::InitDecl(var_decl) => {
                        if var_decl.storage_class != StorageClass::None {
                            return Err(TypecheckError::StorageClassForLoop);
                        }
                        self.typecheck_local_var_declaration(var_decl)?.into()
                    }
                    ForInit::InitExp(Some(expr)) => Some(self.typecheck_and_convert(expr)?).into(),
                    ForInit::InitExp(None) => None.into(),
                });
                let cond = cond
                    .map(|e| {
                        let typed_e = self.typecheck_exp(e)?;
                        if typed_e.get_type().is_scalar() {
                            Ok(typed_e)
                        } else {
                            Err(TypecheckError::ConditionMustBeScalar)
                        }
                    })
                    .transpose()?;
                let incr = incr.map(|e| self.typecheck_exp(e)).transpose()?;
                let body = self.typecheck_statement(body)?;

                Stmt::For(init, cond, incr, body, label)
            }
            Stmt::Switch(expr, body, cases, default_label, label) => {
                let mut typed_expr = self.typecheck_and_convert(expr)?;
                let mut expr_type = typed_expr.get_type().clone();

                if expr_type.is_character() {
                    typed_expr = convert_to(typed_expr, &CType::Int);
                    expr_type = CType::Int;
                }

                if !expr_type.is_integer() {
                    return Err(TypecheckError::MiscError(
                        "Switch expression must be integer",
                        typed_expr.get_line(),
                    ));
                }

                let mut seen_cases = HashSet::new();

                let typechecked_cases = cases
                    .into_iter()
                    .map(|mut case| {
                        if case.value.get_type().is_integer() {
                            case.value = case.value.convert_type(&expr_type);
                            if seen_cases.contains(&case.value) {
                                Err(TypecheckError::MiscError("Duplicate case value", 0))
                            } else {
                                seen_cases.insert(case.value);
                                Ok(case)
                            }
                        } else {
                            Err(TypecheckError::MiscError("Case value must be integer", 0))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                Stmt::Switch(
                    typed_expr,
                    self.typecheck_statement(body)?,
                    typechecked_cases,
                    default_label,
                    label,
                )
            }
            s => s,
        })
    }

    fn typecheck_initializer(
        &mut self,
        init: Initializer,
        target_type: &CType,
    ) -> Result<Initializer> {
        init.type_map(|i, line| self.typecheck_init(i, target_type, line))
    }

    fn typecheck_init(
        &mut self,
        init: Init,
        target_type: &CType,
        line: usize,
    ) -> Result<(Init, CType)> {
        match (target_type, init) {
            (CType::Array(element_type, size), Init::SingleInit(e)) if e.is_string() => {
                if !element_type.is_character() {
                    return Err(TypecheckError::CannotInitializeNonCharacter(line));
                }
                let s = e.unwrap().unwrap_string();
                if s.len() > *size {
                    return Err(TypecheckError::WrongInitializerLength(line));
                }

                let mut expr = Expr::String(s).at_line(line);
                expr.set_type(target_type.clone());

                Ok((
                    Init::SingleInit(expr),
                    CType::Array(element_type.clone(), *size),
                ))
            }
            (CType::Structure(tag), Init::CompoundInit(init_list)) => {
                let struct_def = self
                    .types
                    .get(&tag.value)
                    .expect("Type should be defined")
                    .clone();

                if init_list.len() > struct_def.members.len() {
                    return Err(TypecheckError::MiscError(
                        "Too many elements in structure initializer",
                        line,
                    ));
                }

                let mut index = 0;
                let mut typechecked_inits = vec![];

                for init in init_list {
                    let member_type = &struct_def.members[index].member_type;
                    typechecked_inits.push(self.typecheck_initializer(init, member_type)?);
                    index += 1;
                }

                while index < struct_def.members.len() {
                    typechecked_inits.push(Init::zero_for(
                        &struct_def.members[index].member_type,
                        &self.types,
                    ));
                    index += 1;
                }

                Ok((Init::CompoundInit(typechecked_inits), target_type.clone()))
            }
            (target_type, Init::SingleInit(e)) => {
                let line = e.get_line();
                let typed_e = self.typecheck_and_convert(e)?;

                Ok((
                    Init::SingleInit(convert_by_assignment(typed_e, target_type, line)?),
                    target_type.clone(),
                ))
            }
            (CType::Array(element_type, size), Init::CompoundInit(init_list)) => {
                if init_list.len() > *size {
                    return Err(TypecheckError::WrongInitializerLength(line));
                }
                let mut typechecked_list = init_list
                    .into_iter()
                    .map(|i| self.typecheck_initializer(i, element_type))
                    .collect::<Result<Vec<_>>>()?;
                while typechecked_list.len() < *size {
                    typechecked_list.push(Initializer::zero(element_type));
                }

                Ok((Init::CompoundInit(typechecked_list), target_type.clone()))
            }
            _ => Err(TypecheckError::CannotInitializeScalar(line)),
        }
    }

    fn typecheck_constant_initializer(
        &mut self,
        init: &Initializer,
        var_type: &CType,
    ) -> Result<InitialValue> {
        if !init.is_constant() {
            return Err(TypecheckError::NonConstantInitializer);
        }

        match (var_type, init) {
            (CType::Array(..), init) if init.is_compound() => Ok(self
                .convert_static_compound_initializers(init.unwrap_compound_ref(), var_type)?
                .into()),
            (CType::Array(..), init) if init.is_string() => Ok(convert_string_initializer(
                init.clone().get_string(),
                var_type,
                init.get_line(),
            )?
            .into()),
            (CType::Array(..), _) => Err(TypecheckError::MiscError(
                "Cannot initialize array with single initializer",
                init.get_line(),
            )),
            (CType::Structure(tag), init) if init.is_compound() => {
                let struct_def = self
                    .types
                    .get(&tag.value)
                    .expect("Struct should be defined")
                    .clone();
                let init_list = init.unwrap_compound_ref();
                if init_list.len() > struct_def.members.len() {
                    return Err(TypecheckError::WrongInitializerLength(
                        init_list[0].get_line(),
                    ));
                }
                let mut current_offset = 0;
                let mut static_inits = vec![];
                for (init_elem, member) in init_list.iter().zip(struct_def.members.iter()) {
                    if member.offset != current_offset {
                        static_inits.push(StaticInit::ZeroInit(member.offset - current_offset));
                    }

                    static_inits.extend(
                        self.typecheck_constant_initializer(init_elem, &member.member_type)?
                            .unwrap_initial()
                            .into_iter(),
                    );

                    current_offset = member.offset + member.member_type.size(&self.types);
                }

                if struct_def.size != current_offset {
                    static_inits.push(StaticInit::ZeroInit(struct_def.size - current_offset));
                }

                Ok(static_inits.into())
            }
            (CType::Structure(..), _) => Err(TypecheckError::MiscError(
                "Cannot initialize structure with single initializer",
                init.get_line(),
            )),
            (_, init) if !init.is_single() => {
                Err(TypecheckError::CannotInitializeScalar(init.get_line()))
            }
            (CType::Pointer(element), init) if init.is_string() => {
                if element.is_char() {
                    let s = init.clone().get_string();
                    let const_name = self.generate_constant_name().leak();
                    self.symbols.insert(
                        const_name,
                        SymbolEntry {
                            c_type: CType::Array(Box::from(CType::Char), s.len() + 1),
                            attrs: IdentifierAttrs::Const(StaticInit::StringInit(s, true)),
                        },
                    );

                    Ok(vec![StaticInit::PointerInit(const_name)].into())
                } else {
                    Err(TypecheckError::CannotInitializeNonCharacter(0))
                }
            }
            (_, init) if init.is_string() => {
                Err(TypecheckError::CannotInitializeScalar(init.get_line()))
            }
            _ => {
                let constant = init.clone().get_constant();
                if var_type.is_pointer() && !constant.is_null_pointer_constant() {
                    Err(TypecheckError::MiscError(
                        "Cannot initialize static pointer with non-null constant",
                        init.get_line(),
                    ))
                } else {
                    Ok(StaticInit::from_constant(init.clone().get_constant(), var_type).into())
                }
            }
        }
    }

    fn typecheck_file_scope_var_decl(
        &mut self,
        mut decl: VarDeclaration,
    ) -> Result<VarDeclaration> {
        self.validate_value_type_specifier(&decl.var_type)?;
        if decl.var_type == CType::Void {
            return Err(TypecheckError::NoIncompleteVariables);
        }
        let mut initial_value = match decl.init {
            Some(ref i) => self.typecheck_constant_initializer(i, &decl.var_type)?,
            None => {
                if decl.storage_class == StorageClass::Extern {
                    InitialValue::NoInitializer
                } else {
                    InitialValue::Tentative
                }
            }
        };

        if (decl.init.is_some() || decl.storage_class == StorageClass::Static)
            && !decl.var_type.is_complete(&self.types)
        {
            return Err(TypecheckError::MiscError(
                "Incomplete typed variables cannot be defined, only declared",
                0,
            ));
        }

        let mut global = decl.storage_class != StorageClass::Static;

        if let Some(old_decl) = self.symbols.get(&decl.name.value) {
            if old_decl.c_type.is_function() {
                return Err(TypecheckError::FunctionRedeclaredAsVar);
            }
            if old_decl.c_type != decl.var_type {
                return Err(TypecheckError::TypeRedefined);
            }
            let attrs = old_decl.attrs.unwrap_static_ref();
            if decl.storage_class == StorageClass::Extern {
                global = attrs.global;
            } else if attrs.global != global {
                return Err(TypecheckError::ConflictingLinkage);
            }

            if let old_init @ InitialValue::Initial(_) = &attrs.init {
                if let InitialValue::Initial(_) = initial_value {
                    return Err(TypecheckError::ConflictingDefinitions);
                } else {
                    initial_value = old_init.clone();
                }
            } else if initial_value == InitialValue::NoInitializer
                && attrs.init == InitialValue::Tentative
            {
                initial_value = InitialValue::Tentative;
            }
        }

        self.symbols.insert(
            decl.name.value,
            SymbolEntry {
                c_type: decl.var_type.clone(),
                attrs: IdentifierAttrs::Static(StaticAttr {
                    init: initial_value,
                    global,
                }),
            },
        );

        decl.init = decl.init.map(|mut expr| {
            expr.set_type(decl.var_type.clone());
            expr
        });

        Ok(decl)
    }

    fn typecheck_local_var_declaration(
        &mut self,
        mut decl: VarDeclaration,
    ) -> Result<VarDeclaration> {
        self.validate_value_type_specifier(&decl.var_type)?;
        if !decl.var_type.is_complete(&self.types) {
            return Err(TypecheckError::NoIncompleteVariables);
        }
        if decl.storage_class == StorageClass::Extern {
            if decl.init.is_some() {
                return Err(TypecheckError::LocalExternInit);
            }
            if let Some(old_decl) = self.symbols.get(&decl.name.value) {
                if old_decl.c_type.is_function() {
                    return Err(TypecheckError::FunctionRedeclaredAsVar);
                }
                if old_decl.c_type != decl.var_type {
                    return Err(TypecheckError::TypeRedefined);
                }
            } else {
                self.symbols.insert(
                    decl.name.value,
                    SymbolEntry {
                        c_type: decl.var_type.clone(),
                        attrs: IdentifierAttrs::Static(StaticAttr {
                            init: InitialValue::NoInitializer,
                            global: true,
                        }),
                    },
                );
            }
        } else if decl.storage_class == StorageClass::Static {
            let var_type = decl.var_type.clone();
            let init = match decl.init {
                Some(ref i) => self.typecheck_constant_initializer(i, &var_type)?,
                None => StaticInit::zero(&var_type, &self.types).into(),
            };

            self.symbols.insert(
                decl.name.value,
                SymbolEntry {
                    c_type: var_type.clone(),
                    attrs: IdentifierAttrs::Static(StaticAttr {
                        init,
                        global: false,
                    }),
                },
            );

            decl.init = decl.init.map(|mut init| {
                init.set_type(var_type);
                init
            });
        } else {
            self.symbols.insert(
                decl.name.value,
                SymbolEntry {
                    c_type: decl.var_type.clone(),
                    attrs: IdentifierAttrs::Local,
                },
            );

            decl.init = decl
                .init
                .map(|i| self.typecheck_initializer(i, &decl.var_type))
                .transpose()?;
        }
        Ok(decl)
    }

    fn typecheck_fn_declaration(
        &mut self,
        mut decl: FunctionDeclaration,
        top_level: bool,
    ) -> Result<FunctionDeclaration> {
        self.validate_function_type_specifier(&decl.fun_type)?;
        if decl.fun_type.ret.is_array() {
            return Err(TypecheckError::CannotReturnArray);
        }

        decl.fun_type.params = decl
            .fun_type
            .params
            .into_iter()
            .map(|t| match t {
                CType::Array(element, _size) => CType::Pointer(element),
                t => t,
            })
            .collect();

        if decl.fun_type.params.contains(&CType::Void) {
            return Err(TypecheckError::NoIncompleteVariables);
        }

        let fn_type = decl.fun_type.clone().into();
        let has_body = decl.body.is_some();
        let mut already_defined = false;
        let mut global = decl.storage_class != StorageClass::Static;

        if let Some(old_decl) = self.symbols.get(&decl.name.value) {
            if old_decl.c_type != fn_type {
                return Err(TypecheckError::IncompatibleFunctionDeclarations);
            }
            let attrs = old_decl.attrs.unwrap_fun_ref();
            already_defined = attrs.defined;
            if already_defined && has_body {
                return Err(TypecheckError::MultipleFunctionDefinition);
            }

            if attrs.global && !global {
                return Err(TypecheckError::IncompatibleFunctionDeclarations);
            }
            global = attrs.global;
        }

        if has_body {
            let ret_type = &fn_type.unwrap_function_ref().ret;
            if *ret_type != CType::Void && !ret_type.is_complete(&self.types) {
                return Err(TypecheckError::MiscError(
                    "Cannot define a function with an incomplete return type",
                    0,
                ));
            }
            if !fn_type
                .unwrap_function_ref()
                .params
                .iter()
                .all(|p| p.is_complete(&self.types))
            {
                return Err(TypecheckError::MiscError(
                    "Cannot define a function with an incomplete parameter type",
                    0,
                ));
            }
        }

        self.symbols.insert(
            decl.name.value,
            SymbolEntry {
                c_type: fn_type,
                attrs: IdentifierAttrs::Fun(FunAttr {
                    defined: already_defined || has_body,
                    global,
                }),
            },
        );

        let body = if has_body {
            if !top_level {
                return Err(TypecheckError::NestedFunctionDefinitions);
            }

            for (param, param_type) in decl.params.iter().zip(decl.fun_type.params.iter()) {
                self.symbols.insert(
                    param.value,
                    SymbolEntry {
                        c_type: param_type.clone(),
                        attrs: IdentifierAttrs::Local,
                    },
                );
            }
            self.current_return_type = Some(decl.fun_type.ret.clone());
            Some(self.typecheck_block(decl.body.unwrap())?)
        } else {
            None
        };

        decl.body = body;

        Ok(decl)
    }

    fn validate_struct_definition(&mut self, decl: &StructDeclaration) -> Result<()> {
        if self.types.contains_key(&decl.tag.value) {
            return Err(TypecheckError::MiscError("Duplicate struct definition", 0));
        }

        let mut seen_member_names: HashSet<&'static str> = HashSet::new();

        for member in &decl.members {
            if seen_member_names.contains(&member.name.value) {
                return Err(TypecheckError::MiscError("Duplicate struct member name", 0));
            }
            seen_member_names.insert(member.name.value);

            if !member.member_type.is_valid_struct_member(&self.types) {
                return Err(TypecheckError::MiscError(
                    "Struct member is incomplete type",
                    0,
                ));
            }
        }

        Ok(())
    }

    fn typecheck_struct_declaration(&mut self, decl: &StructDeclaration) -> Result<()> {
        if decl.members.is_empty() {
            return Ok(());
        }

        self.validate_struct_definition(decl)?;

        let mut member_entries = vec![];
        let mut struct_size: usize = 0;
        let mut struct_alignment = 1;

        for member in &decl.members {
            let member_alignment = member.member_type.type_alignment(&self.types);
            let member_offset = struct_size
                + if !struct_size.is_multiple_of(member_alignment) {
                    member_alignment - struct_size % member_alignment
                } else {
                    0
                };
            member_entries.push(MemberEntry {
                name: member.name,
                member_type: member.member_type.clone(),
                offset: member_offset,
            });

            struct_alignment = max(struct_alignment, member_alignment);
            struct_size = member_offset + member.member_type.size(&self.types);
        }

        if !struct_size.is_multiple_of(struct_alignment) {
            struct_size += struct_alignment - struct_size % struct_alignment;
        }
        self.types.insert(
            decl.tag.value,
            StructEntry {
                alignment: struct_alignment,
                size: struct_size,
                members: member_entries,
            },
        );

        Ok(())
    }

    fn typecheck_declaration(&mut self, decl: Declaration, top_level: bool) -> Result<Declaration> {
        Ok(match decl {
            Declaration::Fn(function) => self.typecheck_fn_declaration(function, top_level)?.into(),
            Declaration::Var(var) => if top_level {
                self.typecheck_file_scope_var_decl(var)?
            } else {
                self.typecheck_local_var_declaration(var)?
            }
            .into(),
            Declaration::Struct(struct_def) => {
                self.typecheck_struct_declaration(&struct_def)?;
                struct_def.into()
            }
        })
    }

    fn typecheck_block(&mut self, block: Block) -> Result<Block> {
        block.map(|item| {
            Ok(match item {
                BlockItem::S(stmt) => self.typecheck_statement(stmt)?.into(),
                BlockItem::D(decl) => self.typecheck_declaration(decl, false)?.into(),
            })
        })
    }

    fn typecheck_program(mut self, program: Program) -> Result<(Program, SymbolTable, TypeTable)> {
        let typed_program = program.map(|d| self.typecheck_declaration(d, true))?;

        Ok((typed_program, self.symbols, self.types))
    }
}

pub fn typecheck_program(program: Program) -> Result<(Program, SymbolTable, TypeTable)> {
    Typechecker::new().typecheck_program(program)
}

use derive_more::{Display, From, IsVariant, Unwrap};
use serde::Serialize;
use std::error::Error;

use crate::common::ctype::{CType, FunctionType};
use crate::common::type_table::TypeTable;
use crate::common::{CodeLabel, Constant, Identifier, print_option, print_vec};
use crate::lexer::token::{Token, TokenType};

#[derive(Clone, Debug, Display, From, Serialize)]
#[display("{}", print_vec(declarations, "\n\n"))]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

impl Program {
    pub fn map<E, F: FnMut(Declaration) -> Result<Declaration, E>>(
        self,
        transform: F,
    ) -> Result<Self, E> {
        Ok(self
            .declarations
            .into_iter()
            .map(transform)
            .collect::<Result<Vec<_>, E>>()?
            .into())
    }
}

#[derive(Clone, Debug, Display, Serialize)]
#[display(
    "(declare{storage_class} (fn {name} ({}))\n{})",
    print_vec(params, " "),
    print_option(body)
)]
pub struct FunctionDeclaration {
    pub name: Identifier,
    pub params: Vec<Identifier>,
    pub body: Option<Block>,
    pub storage_class: StorageClass,
    pub fun_type: FunctionType,
}

impl FunctionDeclaration {
    pub fn map<E, F: FnMut(BlockItem) -> Result<BlockItem, E>>(
        self,
        transform: F,
    ) -> Result<Self, E> {
        Ok(FunctionDeclaration {
            name: self.name,
            params: self.params,
            body: self.body.map(|b| b.map(transform)).transpose()?,
            storage_class: self.storage_class,
            fun_type: self.fun_type,
        })
    }

    pub fn map_block<E, F: FnMut(Block) -> Result<Block, E>>(
        self,
        transform: F,
    ) -> Result<Self, E> {
        Ok(FunctionDeclaration {
            name: self.name,
            params: self.params,
            body: self.body.map(transform).transpose()?,
            storage_class: self.storage_class,
            fun_type: self.fun_type,
        })
    }
}

#[derive(Clone, Debug, Display, From, Serialize, Default)]
#[display("{}", print_vec(_0, "\n"))]
pub struct Block(pub Vec<BlockItem>);

impl Block {
    pub fn map<E, F: FnMut(BlockItem) -> Result<BlockItem, E>>(
        self,
        transform: F,
    ) -> Result<Self, E> {
        Ok(self
            .0
            .into_iter()
            .map(transform)
            .collect::<Result<Vec<BlockItem>, E>>()?
            .into())
    }

    pub fn map_stmt<E, F: FnMut(Statement) -> Result<Statement, E>>(
        self,
        mut transform: F,
    ) -> Result<Self, E> {
        self.map(|item| match item {
            BlockItem::S(stmt) => Ok(transform(stmt)?.into()),
            item => Ok(item),
        })
    }
}

#[derive(Clone, Debug, Display, From, Serialize)]
#[serde(untagged)]
pub enum BlockItem {
    S(Statement),
    D(Declaration),
}

#[derive(Clone, Debug, Display, From, Serialize)]
#[serde(tag = "type")]
pub enum Declaration {
    Fn(FunctionDeclaration),
    Var(VarDeclaration),
    Struct(StructDeclaration),
}

impl Declaration {
    pub fn fn_map<E: Error>(
        self,
        mut fn_transform: impl FnMut(FunctionDeclaration) -> Result<FunctionDeclaration, E>,
    ) -> Result<Declaration, E> {
        Ok(match self {
            Declaration::Fn(function) => fn_transform(function)?.into(),
            d => d,
        })
    }
}

#[derive(Clone, Debug, Display, Serialize)]
#[display(
    "(declare{storage_class} (var {var_type} {name}) {})",
    print_option(init)
)]
pub struct VarDeclaration {
    pub name: Identifier,
    pub init: Option<Initializer>,
    pub storage_class: StorageClass,
    pub var_type: CType,
}

#[derive(Clone, Debug, Display, Serialize)]
#[display("(declare struct {tag} {})", if members.is_empty() { "nil".to_owned()} else { print_vec(members, " ")})]
pub struct StructDeclaration {
    pub tag: Identifier,
    pub members: Vec<MemberDeclaration>,
}

#[derive(Clone, Debug, Display, Serialize)]
#[display("({member_type} {name})")]
pub struct MemberDeclaration {
    pub name: Identifier,
    pub member_type: CType,
}

#[derive(Clone, Debug, From, Display, Serialize)]
#[serde(untagged)]
pub enum ForInit {
    InitDecl(VarDeclaration),
    #[display("{}", print_option(_0))]
    InitExp(Option<Expression>),
}

#[derive(Clone, Debug, From, Unwrap, Serialize)]
pub enum Label {
    Plain(CodeLabel),
    Case(Expression),
    Default,
}

impl From<Token> for Label {
    fn from(value: Token) -> Self {
        let label: CodeLabel = value.into();
        label.into()
    }
}

#[derive(Clone, Debug, Display, Serialize)]
#[display("{content}")]
pub struct Statement {
    pub labels: Vec<Label>,
    content: Box<Stmt>,
}

impl Statement {
    pub fn new(content: Stmt, labels: Vec<Label>) -> Self {
        Self {
            content: Box::new(content),
            labels,
        }
    }

    pub fn map<E: Error>(self, transform: impl FnOnce(Stmt) -> Result<Stmt, E>) -> Result<Self, E> {
        Ok(Self {
            labels: self.labels,
            content: Box::from(transform(*self.content)?),
        })
    }

    pub fn visit<T, E: Error>(
        &self,
        mut transform: impl FnMut(&Stmt) -> Result<T, E>,
    ) -> Result<T, E> {
        transform(&self.content)
    }

    pub fn visit_infallible<T>(self, mut transform: impl FnMut(Stmt) -> T) -> T {
        transform(*self.content)
    }
}

impl From<Stmt> for Statement {
    fn from(value: Stmt) -> Self {
        Self {
            labels: vec![],
            content: Box::new(value),
        }
    }
}

#[derive(Clone, Debug, Display, Serialize)]
pub enum Stmt {
    #[display("(return {})", print_option(_0))]
    Return(Option<Expression>),
    Expression(Expression),
    #[display("(if {_0} {_1} {})", print_option(_2))]
    If(Expression, Statement, Option<Statement>),
    #[display("(block {_0})")]
    Compound(Block),
    #[display("(break :{})", print_option(_0))]
    Break(Option<CodeLabel>),
    #[display("(continue :{})", print_option(_0))]
    Continue(Option<CodeLabel>),
    #[display("(while :{} {_0} {_1})", print_option(_2))]
    While(Expression, Statement, Option<CodeLabel>),
    #[display("(do-while :{} {_0} {_1})", print_option(_2))]
    DoWhile(Statement, Expression, Option<CodeLabel>),
    #[display(
        "(for :{} {_0} {} {} {_3})",
        print_option(_4),
        print_option(_1),
        print_option(_2)
    )]
    For(
        Box<ForInit>,
        Option<Expression>,
        Option<Expression>,
        Statement,
        Option<CodeLabel>,
    ),
    #[display("(goto {_0})")]
    Goto(CodeLabel),
    #[display("(switch {_0} {_1})")]
    Switch(
        Expression,
        Statement,
        Vec<CaseInfo>,
        Option<CodeLabel>,
        Option<CodeLabel>,
    ),
    #[display("()")]
    Null,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseInfo {
    pub value: Constant,
    pub label: CodeLabel,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExpressionMetadata {
    pub line: usize,
    pub value_type: Option<CType>,
}

#[derive(Clone, Debug, Serialize, Display)]
#[display("{content}")]
pub struct Initializer {
    metadata: ExpressionMetadata,
    content: Init,
}

impl From<Init> for Initializer {
    fn from(value: Init) -> Self {
        Self {
            metadata: Default::default(),
            content: value,
        }
    }
}

impl Initializer {
    pub fn zero(value_type: &CType) -> Self {
        match value_type {
            CType::Void => unimplemented!(),
            CType::Int => Expr::Constant(Constant::Int(0))
                .with_type(CType::Int)
                .into(),
            CType::Long => Expr::Constant(Constant::Long(0))
                .with_type(CType::Long)
                .into(),
            CType::Unsigned => Expr::Constant(Constant::UInt(0))
                .with_type(CType::Unsigned)
                .into(),
            CType::UnsignedLong => Expr::Constant(Constant::ULong(0))
                .with_type(CType::UnsignedLong)
                .into(),
            CType::Double => Expr::Constant(Constant::Double(0.0))
                .with_type(CType::Double)
                .into(),
            CType::Char => Expr::Constant(Constant::Char(0))
                .with_type(CType::Char)
                .into(),
            CType::SignedChar => Expr::Constant(Constant::Char(0))
                .with_type(CType::SignedChar)
                .into(),
            CType::UnsignedChar => Expr::Constant(Constant::UChar(0))
                .with_type(CType::UnsignedChar)
                .into(),
            CType::Pointer(t) => Expr::Constant(Constant::ULong(0))
                .with_type(CType::Pointer(t.clone()))
                .into(),
            CType::Array(t, size) => Init::CompoundInit(vec![Self::zero(t); *size])
                .with_type(CType::Array(t.clone(), *size)),
            CType::Function(_) => panic!(),
            _ => unimplemented!(),
        }
    }

    pub fn set_type(&mut self, value_type: CType) {
        self.metadata.value_type = Some(value_type);
    }

    pub fn get_type(&self) -> &CType {
        self.metadata.value_type.as_ref().unwrap()
    }

    pub fn get_line(&self) -> usize {
        self.metadata.line
    }

    pub fn set_line(&mut self, line: usize) {
        self.metadata.line = line;
    }

    pub fn type_map<E: Error>(
        mut self,
        mut transform: impl FnMut(Init, usize) -> Result<(Init, CType), E>,
    ) -> Result<Self, E> {
        let (expr, value_type) = transform(self.content, self.metadata.line)?;

        self.metadata.value_type = Some(value_type);
        self.content = expr;

        Ok(self)
    }

    pub fn is_constant(&self) -> bool {
        match self.content {
            Init::SingleInit(ref e) => e.is_constant(),
            Init::CompoundInit(ref inits) => inits.iter().all(|i| i.is_constant()),
        }
    }

    pub fn is_single(&self) -> bool {
        self.content.is_single_init()
    }

    pub fn is_compound(&self) -> bool {
        self.content.is_compound_init()
    }

    pub fn is_string(&self) -> bool {
        self.is_single() && self.content.unwrap_single_init_ref().is_string()
    }

    pub fn get_string(self) -> String {
        self.content.unwrap_single_init().unwrap().unwrap_string()
    }

    pub fn get_constant(self) -> Constant {
        self.content.unwrap_single_init().unwrap().unwrap_constant()
    }

    pub fn unwrap_single(self) -> Expression {
        self.content.unwrap_single_init()
    }

    pub fn unwrap_single_ref(&self) -> &Expression {
        self.content.unwrap_single_init_ref()
    }

    pub fn unwrap(self) -> Init {
        self.content
    }

    pub fn unwrap_compound(self) -> Vec<Initializer> {
        self.content.unwrap_compound_init()
    }

    pub fn unwrap_compound_ref(&self) -> &Vec<Initializer> {
        self.content.unwrap_compound_init_ref()
    }
}

impl From<Expression> for Initializer {
    fn from(value: Expression) -> Self {
        let value_type = value.metadata.value_type.clone();
        let line = value.metadata.line;
        let content: Init = value.into();
        let mut initializer = if let Some(value_type) = value_type {
            content.with_type(value_type)
        } else {
            content.into()
        };
        initializer.metadata.line = line;
        initializer
    }
}

impl From<Vec<Initializer>> for Initializer {
    fn from(value: Vec<Initializer>) -> Self {
        let content: Init = value.into();
        content.into()
    }
}

#[derive(Clone, Debug, Serialize, IsVariant, Unwrap, Display, From)]
#[unwrap(ref)]
pub enum Init {
    SingleInit(Expression),
    #[display("{}", print_vec(_0, " ,"))]
    CompoundInit(Vec<Initializer>),
}

impl Init {
    pub fn zero_for(value_type: &CType, type_table: &TypeTable) -> Initializer {
        match value_type {
            CType::Void => panic!("no void zero"),
            CType::Int => Self::SingleInit(Expr::Constant(Constant::Int(0)).with_type(CType::Int)),
            CType::Long => {
                Self::SingleInit(Expr::Constant(Constant::Long(0)).with_type(CType::Long))
            }
            CType::Short => {
                Self::SingleInit(Expr::Constant(Constant::Short(0)).with_type(CType::Short))
            }
            CType::Unsigned => {
                Self::SingleInit(Expr::Constant(Constant::UInt(0)).with_type(CType::Unsigned))
            }
            CType::UnsignedLong => {
                Self::SingleInit(Expr::Constant(Constant::ULong(0)).with_type(CType::UnsignedLong))
            }
            CType::UnsignedShort => Self::SingleInit(
                Expr::Constant(Constant::UShort(0)).with_type(CType::UnsignedShort),
            ),
            CType::Char => {
                Self::SingleInit(Expr::Constant(Constant::Char(0)).with_type(CType::Char))
            }
            CType::SignedChar => {
                Self::SingleInit(Expr::Constant(Constant::Char(0)).with_type(CType::SignedChar))
            }
            CType::UnsignedChar => {
                Self::SingleInit(Expr::Constant(Constant::UChar(0)).with_type(CType::UnsignedChar))
            }
            CType::Double => {
                Self::SingleInit(Expr::Constant(Constant::Double(0.0)).with_type(CType::Double))
            }
            CType::Pointer(_) => {
                Self::SingleInit(Expr::Constant(Constant::ULong(0)).with_type(value_type.clone()))
            }
            CType::Array(element, count) => {
                Self::CompoundInit(vec![Self::zero_for(element, type_table); *count])
            }

            CType::Function(..) => panic!("No zero function"),
            CType::Structure(tag) => Self::CompoundInit(
                type_table
                    .get(&tag.value)
                    .unwrap()
                    .members
                    .iter()
                    .map(|m| Self::zero_for(&m.member_type, type_table))
                    .collect(),
            ),
        }
        .with_type(value_type.clone())
    }

    pub fn with_type(self, value_type: CType) -> Initializer {
        let mut initializer: Initializer = self.into();
        initializer.metadata.value_type = Some(value_type);
        initializer
    }
}

#[derive(Debug, Clone, Display, Serialize)]
#[display("{content}")]
pub struct Expression {
    metadata: ExpressionMetadata,
    content: Box<Expr>,
}

impl Expression {
    pub fn map<E: Error>(
        self,
        mut transform: impl FnMut(Expr, usize) -> Result<Expr, E>,
    ) -> Result<Self, E> {
        let line = self.metadata.line;
        Ok(Self {
            metadata: self.metadata,
            content: Box::from(transform(*self.content, line)?),
        })
    }

    pub fn type_map<E: Error>(
        mut self,
        mut transform: impl FnMut(Expr, usize) -> Result<(Expr, CType), E>,
    ) -> Result<Self, E> {
        let line = self.metadata.line;

        let (expr, value_type) = transform(*self.content, line)?;

        self.metadata.value_type = Some(value_type);
        self.content = Box::from(expr);

        Ok(self)
    }

    pub fn set_type(&mut self, value_type: CType) {
        self.metadata.value_type = Some(value_type);
    }

    pub fn with_line_from(mut self, token: &Token) -> Self {
        self.metadata.line = token.line;
        self
    }

    pub fn is_constant(&self) -> bool {
        self.content.is_constant() || self.content.is_string()
    }

    pub fn is_var(&self) -> bool {
        self.content.is_var()
    }

    pub fn get_var_name(&self) -> Identifier {
        *self.content.unwrap_var_ref()
    }

    pub fn is_cast(&self) -> bool {
        self.content.is_cast()
    }

    pub fn get_cast_inner(&self) -> Expression {
        self.content.unwrap_cast_ref().1.clone()
    }

    pub fn is_lvalue(&self) -> bool {
        self.content.is_var()
            || self.content.is_dereference()
            || self.content.is_subscript()
            || self.content.is_string()
            || self.content.is_arrow()
            || (self.content.is_dot() && self.content.unwrap_dot_ref().0.is_lvalue())
    }

    pub fn is_string(&self) -> bool {
        self.content.is_string()
    }

    pub fn is_null_pointer_constant(&self) -> bool {
        matches!(
            *self.content,
            Expr::Constant(
                Constant::Int(0) | Constant::Long(0) | Constant::UInt(0) | Constant::ULong(0)
            )
        )
    }

    pub fn get_type(&self) -> &CType {
        self.metadata.value_type.as_ref().unwrap()
    }

    pub fn get_line(&self) -> usize {
        self.metadata.line
    }

    pub fn unwrap(self) -> Expr {
        *self.content
    }
}

impl AsRef<Expr> for Expression {
    fn as_ref(&self) -> &Expr {
        &self.content
    }
}

impl From<Expr> for Expression {
    fn from(value: Expr) -> Self {
        Self {
            metadata: Default::default(),
            content: Box::from(value),
        }
    }
}

#[derive(Clone, Debug, Display, IsVariant, Unwrap, Serialize)]
#[unwrap(ref)]
pub enum Expr {
    Constant(Constant),
    String(String),
    Var(Identifier),
    #[display("((as {_0}) {_1})")]
    Cast(CType, Expression),
    #[display("({_0} {_1})")]
    Unary(UnaryOperator, Expression),
    #[display("({_0} {_1} {_2})")]
    Binary(BinaryOperator, Expression, Expression),
    #[display("(postfix{_0} {_1})")]
    Postfix(PostfixOperator, Expression),
    #[display("(assign {_0} {_1})")]
    Assignment(Expression, Expression),
    #[display("(assign{_0} {_1} {_2})")]
    CompoundAssignment(BinaryOperator, Expression, Expression, Option<CType>),
    #[display("(conditional {_0} {_1} {_2})")]
    Conditional(Expression, Expression, Expression),
    #[display("({_0} {})", print_vec(_1, " "))]
    FunctionCall(Identifier, Vec<Expression>),
    #[display("(dereference {_0})")]
    Dereference(Expression),
    #[display("(addr {_0})")]
    AddrOf(Expression),
    #[display("(subscript {_0} {_1})")]
    Subscript(Expression, Expression),
    #[display("(sizeof {_0})")]
    SizeOf(Expression),
    #[display("(sizeof type {_0})")]
    SizeOfT(CType),
    #[display("(. {_0} {_1})")]
    Dot(Expression, Identifier),
    #[display("(-> {_0} {_1})")]
    Arrow(Expression, Identifier),
}

impl Expr {
    pub fn at_line(self, line: usize) -> Expression {
        let mut expression: Expression = self.into();
        expression.metadata.line = line;
        expression
    }

    pub fn with_type(self, value_type: CType) -> Expression {
        let mut expression: Expression = self.into();
        expression.set_type(value_type);
        expression
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Display, Serialize)]
pub enum UnaryOperator {
    #[display("~")]
    Complement,
    #[display("-")]
    Negate,
    #[display("!")]
    Not,
    #[display("++")]
    PreIncrement,
    #[display("--")]
    PreDecrement,
}

impl UnaryOperator {
    pub fn can_apply_to_pointer(&self) -> bool {
        matches!(self, Self::Not | Self::PreIncrement | Self::PreDecrement)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Display, Serialize)]
pub enum BinaryOperator {
    #[display("+")]
    Add,
    #[display("-")]
    Subtract,
    #[display("*")]
    Multiply,
    #[display("/")]
    Divide,
    #[display("%")]
    Remainder,
    #[display("&")]
    BitwiseAnd,
    #[display("&&")]
    And,
    #[display("|")]
    BitwiseOr,
    #[display("||")]
    Or,
    #[display("^")]
    BitwiseXor,
    #[display("==")]
    Equal,
    #[display("!=")]
    NotEqual,
    #[display("<<")]
    LeftShift,
    #[display(">>")]
    RightShift,
    #[display("<")]
    LessThan,
    #[display("<=")]
    LessOrEqual,
    #[display(">")]
    GreaterThan,
    #[display(">=")]
    GreaterOrEqual,
}

impl BinaryOperator {
    pub fn is_arithmetic(&self) -> bool {
        use BinaryOperator::*;
        matches!(
            self,
            Add | Subtract
                | Multiply
                | Divide
                | Remainder
                | BitwiseAnd
                | BitwiseOr
                | BitwiseXor
                | LeftShift
                | RightShift
        )
    }

    pub fn is_shift(&self) -> bool {
        use BinaryOperator::*;
        matches!(self, LeftShift | RightShift)
    }

    pub fn is_bitwise(&self) -> bool {
        use BinaryOperator::*;
        matches!(
            self,
            LeftShift | RightShift | BitwiseAnd | BitwiseOr | BitwiseXor
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Display, Serialize)]
pub enum PostfixOperator {
    #[display("++")]
    Increment,
    #[display("--")]
    Decrement,
}

#[derive(Clone, Copy, PartialEq, Debug, Display, Serialize)]
pub enum StorageClass {
    #[display(":static")]
    Static,
    #[display(":extern")]
    Extern,
    #[display("")]
    None,
}

impl From<TokenType> for StorageClass {
    fn from(value: TokenType) -> Self {
        match value {
            TokenType::StaticKeyword => Self::Static,
            TokenType::ExternKeyword => Self::Extern,
            _ => unreachable!(),
        }
    }
}

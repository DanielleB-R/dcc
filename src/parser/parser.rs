use std::{
    collections::{HashMap, VecDeque},
    sync::LazyLock,
};

use super::ast::*;
use super::declarator_ast::{AbstractDeclarator, Declarator};
use super::precedence::{BINARY_PRECEDENCES, Precedence};
use crate::common::{CType, Constant, char_escape::unescape};
use crate::errors::ParserError;
use crate::lexer::token::{
    Token,
    TokenType::{self, *},
};

static COMPOUND_ASSIGNMENT_OPERATORS: LazyLock<HashMap<TokenType, BinaryOperator>> =
    LazyLock::new(|| {
        [
            (PlusEqual, BinaryOperator::Add),
            (MinusEqual, BinaryOperator::Subtract),
            (StarEqual, BinaryOperator::Multiply),
            (SlashEqual, BinaryOperator::Divide),
            (PercentEqual, BinaryOperator::Remainder),
            (AndEqual, BinaryOperator::BitwiseAnd),
            (PipeEqual, BinaryOperator::BitwiseOr),
            (HatEqual, BinaryOperator::BitwiseXor),
            (LessLessEqual, BinaryOperator::LeftShift),
            (GreaterGreaterEqual, BinaryOperator::RightShift),
        ]
        .into()
    });

pub type Result<T> = std::result::Result<T, ParserError>;

pub(crate) struct Parser {
    tokens: VecDeque<Token>,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens: tokens.into(),
        }
    }

    pub(crate) fn take(&mut self) -> Token {
        self.tokens.pop_front().unwrap()
    }

    pub(crate) fn peek(&self) -> &Token {
        &self.tokens[0]
    }

    pub(crate) fn peek_next(&self) -> &Token {
        &self.tokens[1]
    }

    pub(crate) fn seeing(&self, token_type: TokenType) -> bool {
        self.peek().token_type == token_type
    }

    pub(crate) fn expect(&mut self, token_type: TokenType) -> Result<Token> {
        let token = self.take();
        if token.token_type == token_type {
            Ok(token)
        } else {
            Err(ParserError::UnexpectedToken {
                expected: token_type,
                found: token.token_type,
                line: token.line,
            })
        }
    }

    pub(crate) fn list_of<T>(
        &mut self,
        mut action: impl FnMut(&mut Self) -> Result<T>,
        delimiter: TokenType,
        terminator: TokenType,
    ) -> Result<Vec<T>> {
        let mut result = vec![action(self)?];

        while self.peek().token_type == delimiter {
            self.take();
            result.push(action(self)?);
        }

        self.expect(terminator)?;
        Ok(result)
    }

    fn repeat_until_terminator(
        &mut self,
        terminator: TokenType,
        mut action: impl FnMut(&mut Self) -> Result<()>,
    ) -> Result<()> {
        while !self.seeing(terminator) {
            action(self)?;
        }

        self.take();

        Ok(())
    }

    fn unary_operator(&mut self) -> Result<UnaryOperator> {
        use UnaryOperator::*;

        let token = self.take();
        match token.token_type {
            Tilde => Ok(Complement),
            Minus => Ok(Negate),
            Bang => Ok(Not),
            PlusPlus => Ok(PreIncrement),
            MinusMinus => Ok(PreDecrement),
            t => Err(ParserError::InvalidUnaryOperator(t, token.line)),
        }
    }

    fn binary_operator(&mut self) -> Result<BinaryOperator> {
        use BinaryOperator::*;
        let token = self.take();
        match token.token_type {
            Plus => Ok(Add),
            Minus => Ok(Subtract),
            Star => Ok(Multiply),
            Slash => Ok(Divide),
            Percent => Ok(Remainder),
            TokenType::And => Ok(BitwiseAnd),
            Pipe => Ok(BitwiseOr),
            Hat => Ok(BitwiseXor),
            LessLess => Ok(LeftShift),
            Less => Ok(LessThan),
            LessEqual => Ok(LessOrEqual),
            GreaterGreater => Ok(RightShift),
            Greater => Ok(GreaterThan),
            GreaterEqual => Ok(GreaterOrEqual),
            EqualEqual => Ok(Equal),
            BangEqual => Ok(NotEqual),
            AndAnd => Ok(And),
            PipePipe => Ok(Or),
            t => Err(ParserError::InvalidBinaryOperator(t, token.line)),
        }
    }

    fn argument_list(&mut self) -> Result<Vec<Expression>> {
        let mut args = vec![];
        if self.seeing(CloseParen) {
            return Ok(args);
        }

        args.push(self.expression(Precedence::Minimum)?);

        while self.seeing(Comma) {
            self.take();
            args.push(self.expression(Precedence::Minimum)?);
        }

        Ok(args)
    }

    pub(crate) fn constant(&mut self) -> Result<Constant> {
        use Constant::*;
        let token = self.take();

        if token.token_type == CharConstant {
            let char_str = unescape(&token.value.unwrap());
            if char_str.len() != 1 {
                panic!("Character literal not one character {}", char_str);
            }
            return Ok(Int(char_str.as_bytes()[0] as i32));
        }

        if token.token_type == FloatingPointConstant {
            return Ok(Double(token.value.unwrap().parse().unwrap()));
        }

        let value: u64 = token
            .value
            .unwrap()
            .parse()
            .map_err(|_| ParserError::ConstantTooLarge(token.line))?;

        match token.token_type {
            UnsignedLongConstant => Ok(ULong(value)),
            UnsignedConstant => {
                if value <= u32::MAX as u64 {
                    Ok(UInt(value as u32))
                } else {
                    Ok(ULong(value))
                }
            }
            t @ (IntConstant | LongConstant) => {
                if value > i64::MAX as u64 {
                    Err(ParserError::ConstantTooLarge(token.line))
                } else if t == IntConstant && value <= i32::MAX as u64 {
                    Ok(Int(value as i32))
                } else {
                    Ok(Long(value as i64))
                }
            }
            _ => unreachable!(),
        }
    }

    fn type_name(&mut self) -> Result<CType> {
        let base_type = self.consume_and_parse_type()?;

        let abstract_declarator = AbstractDeclarator::parse(self)?;
        Ok(abstract_declarator.process(base_type))
    }

    fn primary(&mut self) -> Result<Expression> {
        let next_token = self.peek();
        let line = next_token.line;
        match next_token.token_type {
            t if t.is_constant() => Ok(Expr::Constant(self.constant()?).at_line(line)),
            StringLiteral => {
                let mut literal = String::new();

                while self.seeing(StringLiteral) {
                    let token = self.take();
                    let contents = unescape(&token.value.unwrap());

                    literal.push_str(&contents);
                }
                Ok(Expr::String(literal).at_line(line))
            }
            OpenParen => {
                self.take();

                let inner = self.expression(Precedence::Minimum)?;
                self.expect(CloseParen)?;
                Ok(inner)
            }
            Identifier => {
                let name = self.take();

                if self.seeing(OpenParen) {
                    self.take();
                    let args = self.argument_list()?;

                    self.expect(CloseParen)?;

                    Ok(Expr::FunctionCall(name.into(), args).at_line(line))
                } else {
                    Ok(Expr::Var(name.into()).at_line(line))
                }
            }
            _ => Err(ParserError::MalformedExpression(next_token.line)),
        }
    }

    fn postfix(&mut self) -> Result<Expression> {
        let mut expr = self.primary()?;
        loop {
            match self.peek().token_type {
                PlusPlus | MinusMinus => {
                    let token = self.take();
                    expr = Expr::Postfix(
                        if token.token_type == PlusPlus {
                            PostfixOperator::Increment
                        } else {
                            PostfixOperator::Decrement
                        },
                        expr,
                    )
                    .at_line(token.line);
                }
                OpenBracket => {
                    let token = self.take();
                    let subscript = self.expression(Precedence::Minimum)?;
                    expr = Expr::Subscript(expr, subscript).at_line(token.line);
                    self.expect(CloseBracket)?;
                }
                Dot => {
                    let token = self.take();
                    let name = self.expect(Identifier)?;
                    expr = Expr::Dot(expr, name.into()).at_line(token.line);
                }
                Arrow => {
                    let token = self.take();
                    let name = self.expect(Identifier)?;
                    expr = Expr::Arrow(expr, name.into()).at_line(token.line);
                }
                _ => {
                    break;
                }
            }
        }
        Ok(expr)
    }

    fn factor(&mut self) -> Result<Expression> {
        let next_token = self.peek();
        let line = next_token.line;
        match next_token.token_type {
            Tilde | Minus | Bang | PlusPlus | MinusMinus => {
                let operator = self.unary_operator()?;
                let inner = self.cast_exp()?;
                Ok(Expr::Unary(operator, inner).at_line(line))
            }
            Star => {
                self.take();
                let inner = self.cast_exp()?;
                Ok(Expr::Dereference(inner).at_line(line))
            }
            And => {
                self.take();
                let inner = self.cast_exp()?;
                Ok(Expr::AddrOf(inner).at_line(line))
            }
            SizeofKeyword => {
                self.take();
                if self.seeing(OpenParen) && self.peek_next().token_type.is_type() {
                    self.take();
                    let size_type = self.type_name()?;
                    self.expect(CloseParen)?;
                    Ok(Expr::SizeOfT(size_type).at_line(line))
                } else {
                    let operand = self.factor()?;
                    Ok(Expr::SizeOf(operand).at_line(line))
                }
            }
            _ => self.postfix(),
        }
    }

    fn cast_exp(&mut self) -> Result<Expression> {
        let next_token = self.peek();
        let line = next_token.line;

        if next_token.token_type == OpenParen && self.peek_next().token_type.is_type() {
            self.take();
            let cast_type = self.type_name()?;
            self.expect(CloseParen)?;

            let cast_subject = self.cast_exp()?;
            Ok(Expr::Cast(cast_type, cast_subject).at_line(line))
        } else {
            self.factor()
        }
    }

    fn conditional_middle(&mut self) -> Result<Expression> {
        self.take();
        let expression = self.expression(Precedence::Minimum)?;
        self.expect(Colon)?;

        Ok(expression)
    }

    fn expression(&mut self, min_prec: Precedence) -> Result<Expression> {
        let mut left = self.cast_exp()?;
        let mut next_token = self.peek();

        while let Some(precedence) = BINARY_PRECEDENCES.get(&next_token.token_type).copied() {
            if precedence < min_prec {
                break;
            }
            match next_token.token_type {
                Equal => {
                    let token = self.take();
                    let right = self.expression(precedence)?;
                    left = Expr::Assignment(left, right).at_line(token.line);
                }
                Question => {
                    let line = next_token.line;
                    let middle = self.conditional_middle()?;
                    let right = self.expression(precedence)?;
                    left = Expr::Conditional(left, middle, right).at_line(line);
                }
                t if COMPOUND_ASSIGNMENT_OPERATORS.contains_key(&t) => {
                    let operator = COMPOUND_ASSIGNMENT_OPERATORS.get(&t).copied().unwrap();
                    let token = self.take();
                    let right = self.expression(precedence)?;
                    left =
                        Expr::CompoundAssignment(operator, left, right, None).at_line(token.line);
                }
                _ => {
                    let line = next_token.line;
                    let operator = self.binary_operator()?;
                    let right = self.expression(precedence.increment())?;
                    left = Expr::Binary(operator, left, right).at_line(line);
                }
            }

            next_token = self.peek();
        }
        Ok(left)
    }

    fn optional_expression(
        &mut self,
        precedence: Precedence,
        delimiter: TokenType,
    ) -> Result<Option<Expression>> {
        if self.seeing(delimiter) {
            self.take();
            return Ok(None);
        }

        let expr = self.expression(precedence)?;
        self.expect(delimiter)?;

        Ok(Some(expr))
    }

    fn for_init(&mut self) -> Result<ForInit> {
        Ok(match self.peek().token_type {
            t if t.is_specifier() => match self.declaration()? {
                Declaration::Var(var) => var,
                _ => Err(ParserError::ForFunctionDecl(self.peek().line))?,
            }
            .into(),
            _ => self
                .optional_expression(Precedence::Minimum, Semicolon)?
                .into(),
        })
    }

    fn statement(&mut self) -> Result<Statement> {
        let mut labels = vec![];

        loop {
            match self.peek().token_type {
                Identifier => {
                    if self.peek_next().token_type != Colon {
                        break;
                    }
                    let l = self.take();
                    self.take();
                    labels.push(l.into())
                }
                CaseKeyword => {
                    self.take();
                    let value = self.expression(Precedence::Minimum)?;
                    self.expect(Colon)?;

                    labels.push(value.into());
                }
                DefaultKeyword => {
                    self.take();
                    self.expect(Colon)?;
                    labels.push(Label::Default);
                }
                _ => {
                    break;
                }
            }
        }

        let stmt = match self.peek().token_type {
            ReturnKeyword => {
                self.take();
                let return_value = self.optional_expression(Precedence::Minimum, Semicolon)?;
                Stmt::Return(return_value)
            }
            IfKeyword => {
                self.take();
                self.expect(OpenParen)?;
                let condition = self.expression(Precedence::Minimum)?;
                self.expect(CloseParen)?;

                let then_stmt = self.statement()?;

                let else_stmt = if self.peek().token_type == ElseKeyword {
                    self.take();
                    Some(self.statement()?)
                } else {
                    None
                };

                Stmt::If(condition, then_stmt, else_stmt)
            }
            BreakKeyword => {
                self.take();
                self.expect(Semicolon)?;
                Stmt::Break(None)
            }
            ContinueKeyword => {
                self.take();
                self.expect(Semicolon)?;
                Stmt::Continue(None)
            }
            DoKeyword => {
                self.take();

                let body = self.statement()?;

                self.expect(WhileKeyword)?;
                self.expect(OpenParen)?;
                let condition = self.expression(Precedence::Minimum)?;
                self.expect(CloseParen)?;
                self.expect(Semicolon)?;

                Stmt::DoWhile(body, condition, None)
            }
            ForKeyword => {
                self.take();
                self.expect(OpenParen)?;
                let initializer = self.for_init()?;
                let condition = self.optional_expression(Precedence::Minimum, Semicolon)?;
                let increment = self.optional_expression(Precedence::Minimum, CloseParen)?;

                let body = self.statement()?;

                Stmt::For(Box::new(initializer), condition, increment, body, None)
            }
            WhileKeyword => {
                self.take();
                self.expect(OpenParen)?;
                let condition = self.expression(Precedence::Minimum)?;
                self.expect(CloseParen)?;
                let body = self.statement()?;

                Stmt::While(condition, body, None)
            }
            // compound statement!
            OpenBrace => {
                self.take();

                Stmt::Compound(self.block()?)
            }
            GotoKeyword => {
                self.take();

                let label = self.expect(Identifier)?;
                self.expect(Semicolon)?;

                Stmt::Goto(label.into())
            }
            SwitchKeyword => {
                self.take();

                self.expect(OpenParen)?;
                let subject = self.expression(Precedence::Minimum)?;
                self.expect(CloseParen)?;

                let body = self.statement()?;

                Stmt::Switch(subject, body, vec![], None, None)
            }
            // expression statement!
            _ => self
                .optional_expression(Precedence::Minimum, Semicolon)?
                .map(Stmt::Expression)
                .unwrap_or(Stmt::Null),
        };

        Ok(Statement::new(stmt, labels))
    }

    fn parse_type(&mut self, types: &[Token]) -> Result<CType> {
        let mut token_types: Vec<_> = types.iter().map(|token| token.token_type).collect();
        token_types.sort();

        match token_types[..] {
            [Identifier, StructKeyword] => {
                if types[1].token_type == Identifier {
                    Ok(CType::Structure(types[1].clone().into()))
                } else {
                    Err(ParserError::InvalidTypeSpecifier(self.peek().line))
                }
            }
            [VoidKeyword] => Ok(CType::Void),
            [DoubleKeyword] => Ok(CType::Double),
            [IntKeyword] | [IntKeyword, SignedKeyword] | [SignedKeyword] => Ok(CType::Int),
            [IntKeyword, LongKeyword, SignedKeyword]
            | [IntKeyword, LongKeyword]
            | [LongKeyword, SignedKeyword]
            | [LongKeyword] => Ok(CType::Long),
            [IntKeyword, ShortKeyword, SignedKeyword]
            | [IntKeyword, ShortKeyword]
            | [ShortKeyword, SignedKeyword]
            | [ShortKeyword] => Ok(CType::Short),
            [IntKeyword, UnsignedKeyword] | [UnsignedKeyword] => Ok(CType::Unsigned),
            [IntKeyword, LongKeyword, UnsignedKeyword] | [LongKeyword, UnsignedKeyword] => {
                Ok(CType::UnsignedLong)
            }
            [IntKeyword, ShortKeyword, UnsignedKeyword] | [ShortKeyword, UnsignedKeyword] => {
                Ok(CType::UnsignedShort)
            }
            [CharKeyword] => Ok(CType::Char),
            [CharKeyword, SignedKeyword] => Ok(CType::SignedChar),
            [CharKeyword, UnsignedKeyword] => Ok(CType::UnsignedChar),
            _ => Err(ParserError::InvalidTypeSpecifier(self.peek().line)),
        }
    }

    pub(crate) fn consume_and_parse_type(&mut self) -> Result<CType> {
        let mut type_specifiers = vec![];
        while self.peek().token_type.is_type() {
            if self.peek().token_type == StructKeyword {
                type_specifiers.push(self.take());
                type_specifiers.push(self.expect(Identifier)?);
            } else {
                type_specifiers.push(self.take());
            }
        }

        self.parse_type(&type_specifiers)
    }

    fn type_and_storage_class(
        &mut self,
        specifier_list: Vec<Token>,
    ) -> Result<(CType, StorageClass)> {
        let mut types = vec![];
        let mut storage_classes = vec![];
        for specifier in specifier_list {
            if specifier.token_type.is_type() || specifier.token_type == Identifier {
                types.push(specifier);
            } else {
                storage_classes.push(specifier);
            }
        }

        let value_type = self.parse_type(&types)?;

        let storage_class = match storage_classes.len() {
            0 => StorageClass::None,
            1 => storage_classes[0].token_type.into(),
            _ => {
                return Err(ParserError::InvalidStorageSpecifier(self.peek().line));
            }
        };

        Ok((value_type, storage_class))
    }

    fn initializer(&mut self) -> Result<Initializer> {
        if self.peek().token_type == OpenBrace {
            self.take();

            let mut initializers = vec![self.initializer()?];

            while self.seeing(Comma) {
                self.take();
                if self.seeing(CloseBrace) {
                    break;
                }
                initializers.push(self.initializer()?);
            }

            self.expect(CloseBrace)?;

            let line = initializers[0].get_line();
            let mut initializer: Initializer = Init::CompoundInit(initializers).into();
            initializer.set_line(line);
            Ok(initializer)
        } else {
            Ok(self.expression(Precedence::Minimum)?.into())
        }
    }

    fn declaration(&mut self) -> Result<Declaration> {
        let mut specifiers = vec![];

        while self.peek().is_specifier() {
            let token_type = self.peek().token_type;
            specifiers.push(self.take());
            if token_type == StructKeyword {
                specifiers.push(self.expect(Identifier)?);
            }
        }

        let (value_type, storage_class) = self.type_and_storage_class(specifiers)?;

        if value_type.is_structure() && (self.seeing(Semicolon) || self.seeing(OpenBrace)) {
            let tag = value_type.unwrap_structure();
            let mut members = vec![];
            if self.seeing(Semicolon) {
                self.take();
                return Ok(StructDeclaration { tag, members }.into());
            }

            self.expect(OpenBrace)?;

            while self.peek().token_type != CloseBrace {
                let value_type = self.consume_and_parse_type()?;
                let declarator = Declarator::parse(self)?;

                let (member_name, member_type, _) = declarator.process(value_type)?;

                if member_type.is_function() {
                    return Err(ParserError::NoFunctionMember);
                }

                self.expect(Semicolon)?;

                members.push(MemberDeclaration {
                    name: member_name,
                    member_type,
                });
            }

            self.take();
            self.expect(Semicolon)?;
            return if members.is_empty() {
                Err(ParserError::BadStructDefinition)
            } else {
                Ok(StructDeclaration { tag, members }.into())
            };
        }

        let declarator = Declarator::parse(self)?;

        let (name, decl_type, params) = declarator.process(value_type)?;

        match decl_type {
            CType::Function(fun_type) => {
                let body = if self.seeing(OpenBrace) {
                    self.take();
                    Some(self.block()?)
                } else {
                    self.expect(Semicolon)?;
                    None
                };

                Ok(FunctionDeclaration {
                    name,
                    params,
                    body,
                    storage_class,
                    fun_type: *fun_type,
                }
                .into())
            }
            var_type => match self.peek().token_type {
                Equal => {
                    self.take();
                    let init = self.initializer()?;
                    self.expect(Semicolon)?;

                    Ok(VarDeclaration {
                        name,
                        init: Some(init),
                        storage_class,
                        var_type,
                    }
                    .into())
                }
                Semicolon => {
                    self.take();
                    Ok(VarDeclaration {
                        name,
                        init: None,
                        storage_class,
                        var_type,
                    }
                    .into())
                }
                _ => {
                    let next_token = self.take();
                    Err(ParserError::MalformedDeclaration(
                        next_token.token_type,
                        next_token.line,
                        next_token.location,
                    ))
                }
            },
        }
    }

    fn block_item(&mut self) -> Result<BlockItem> {
        Ok(if self.peek().is_specifier() {
            self.declaration()?.into()
        } else {
            self.statement()?.into()
        })
    }

    fn block(&mut self) -> Result<Block> {
        let mut body = vec![];

        self.repeat_until_terminator(CloseBrace, |parser| {
            body.push(parser.block_item()?);
            Ok(())
        })?;

        Ok(body.into())
    }

    fn program(mut self) -> Result<Program> {
        let mut declarations = vec![];

        self.repeat_until_terminator(EOF, |parser| {
            parser.declaration().map(|decl| declarations.push(decl))
        })?;

        Ok(declarations.into())
    }
}

pub fn parse_tokens(tokens: Vec<Token>) -> Result<Program> {
    Parser::new(tokens).program()
}

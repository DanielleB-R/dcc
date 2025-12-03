use crate::lexer::token::TokenType;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LexerError {
    #[error("No matching token at char {0} ({1})")]
    NoToken(usize, char),
}

#[derive(Clone, Error, PartialEq, Eq, Debug)]
pub enum ParserError {
    #[error("Expected {expected} and found {found} on line {line}")]
    UnexpectedToken {
        expected: TokenType,
        found: TokenType,
        line: usize,
    },
    #[error("Invalid unary operator {0} at line {1}")]
    InvalidUnaryOperator(TokenType, usize),
    #[error("Invalid binary operator {0} at line {1}")]
    InvalidBinaryOperator(TokenType, usize),

    #[error("Malformed expression at {0}")]
    MalformedExpression(usize),

    #[error("Invalid function argument {0} at {1}")]
    InvalidFunctionArgument(TokenType, usize),

    #[error("Unexpected {0}, expected Equal, Semicolon at line {1} char {2}")]
    MalformedDeclaration(TokenType, usize, usize),

    #[error("No function declarations in for statement at line {0}")]
    ForFunctionDecl(usize),

    #[error("Invalid type specifier at line {0}")]
    InvalidTypeSpecifier(usize),

    #[error("Invalid storage class specifier at line {0}")]
    InvalidStorageSpecifier(usize),

    #[error("Constant too large for integer type at line {0}")]
    ConstantTooLarge(usize),

    #[error("Bad declarator at line {0}")]
    BadDeclarator(usize),

    #[error("Function types not permitted as parameters")]
    NoFunctionParams,

    #[error("Can't apply additional type derivations to a function type")]
    NoFunctionDerivations,

    #[error("Struct definition must have at least one member")]
    BadStructDefinition,

    #[error("No function members in struct definitions")]
    NoFunctionMember,
}

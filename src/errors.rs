use std::io;

use thiserror::Error;

use crate::common::CType;
use crate::lexer::token::TokenType;

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

#[derive(Clone, Debug, Error)]
pub enum SemanticAnalysisError {
    #[error("Duplicate variable declaration!")]
    DuplicateDeclaration,

    #[error("Undeclared variable on line {0} at {1}!")]
    UndeclaredVariable(usize, usize),

    #[error("Undeclared function on line {0}!")]
    UndeclaredFunction(usize),

    #[error("Break statement outside loop!")]
    BareBreak,

    #[error("Continue statement outside loop!")]
    BareContinue,

    #[error("Cannot declare static function in block!")]
    StaticFunctionInBlock,

    #[error("Cannot use undeclared struct")]
    UndeclaredStruct,

    #[error("Labels in a function must be unique")]
    DuplicateLabel,

    #[error("Label not found")]
    UnknownLabel,

    #[error("Non-constant case on line {0}")]
    NonConstantCase(usize),

    #[error("Duplicate case on line {0}")]
    DuplicateCase(usize),

    #[error("Duplicate default for switch")]
    DuplicateDefault,

    #[error("Switch label outside switch statement")]
    BareSwitchLabel,
}

#[derive(Clone, Debug, Error)]
pub enum TypecheckError {
    #[error("Incompatible function declarations")]
    IncompatibleFunctionDeclarations,

    #[error("Function is defined more than once")]
    MultipleFunctionDefinition,

    #[error("Variable is not callable on line {0}")]
    CalledNonCallable(usize),

    #[error("Function called with wrong arguments on line {0}")]
    NonMatchingArguments(usize),

    #[error("Function name used as variable on line {0}")]
    ValueOfCallable(usize),

    #[error("Function defined inside function definition")]
    NestedFunctionDefinitions,

    #[error("Non-constant initializer")]
    NonConstantInitializer,

    #[error("Function redeclared as variable")]
    FunctionRedeclaredAsVar,

    #[error("Conflicting variable linkage")]
    ConflictingLinkage,

    #[error("Conflicting file scope variable definitions")]
    ConflictingDefinitions,

    #[error("Initializer on local extern variable declaration")]
    LocalExternInit,

    #[error("Storage class on for loop variable is invalid")]
    StorageClassForLoop,

    #[error("Variable redefined with different type")]
    TypeRedefined,

    #[error("Cannot use bitwise operations on doubles on line {0}")]
    BitwiseOpOnDouble(usize),

    #[error("Cannot dereference non-pointer on line {0}")]
    NonPointerDereference(usize),

    #[error("Can't take the address of a non-lvalue on line {0}!")]
    NonLvalueAddress(usize),

    #[error("Expressions have incompatible types ({0} and {1}) on line {2}")]
    IncompatibleTypes(CType, CType, usize),

    #[error("Cannot cast between double and pointer on line {0}")]
    PointerDoubleCast(usize),

    #[error("Cannot cast to non-scalar non-void type on line {0}")]
    NonScalarCast(usize),

    #[error("Cannot cast non-scalar expression to scalar type on line {0}")]
    CastNonScalar(usize),

    #[error("Cannot compare two different pointer types on line {0}")]
    BadComparison(usize),

    #[error("Cannot perform arithmetic operation on pointer on line {0}")]
    BadPointerArithmetic(usize),

    #[error("Cannot subscript non-pointer type or with non-integer type on line {0}")]
    BadSubscript(usize),

    #[error("Cannot assign to non-lvalue on line {0}")]
    NonLvalueAssignment(usize),

    #[error("Cannot return an array type from a function")]
    CannotReturnArray,

    #[error("Wrong number of identifiers in initializer on line {0}")]
    WrongInitializerLength(usize),

    #[error("Cannot initialize scalar with compound initializer on line {0}")]
    CannotInitializeScalar(usize),

    #[error("Cannot initialize a non-character type with a string literal on line {0}")]
    CannotInitializeNonCharacter(usize),

    #[error("Void function must have no value in return statements")]
    CannotReturnFromVoid,

    #[error("Non-void function must have value in return statements")]
    MustReturnFromNonVoid,

    #[error("Logical operators only apply to scalar expressions on line {0}")]
    LogicalRequiresScalar(usize),

    #[error("Array must be of complete type")]
    ArrayOfIncomplete,

    #[error("No incomplete variables or parameters")]
    NoIncompleteVariables,

    #[error("No dereferencing void pointers")]
    NoDereferenceToVoid,

    #[error("Condition must be scalar")]
    ConditionMustBeScalar,

    #[error("Cannot convert branches of conditional to common type on line {0}")]
    CannotConvertBranches(usize),

    #[error("Can only increment or decrement numbers and pointers")]
    BadIncrement(usize),

    #[error("{0} on line {1}")]
    MiscError(&'static str, usize),
}

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Error reading source {0}")]
    SourceReadError(#[from] io::Error),

    #[error("Error lexing source {0}")]
    LexError(#[from] LexerError),

    #[error("Error parsing source {0}")]
    ParseError(#[from] ParserError),

    #[error("Error resolving variables or labelling loops {0}")]
    SemanticError(#[from] SemanticAnalysisError),

    #[error("Error checking types {0}")]
    TypeError(#[from] TypecheckError),
}

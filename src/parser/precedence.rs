use std::{collections::HashMap, sync::LazyLock};

use crate::lexer::token::TokenType::{self, *};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precedence {
    Minimum,
    Assignment,
    Conditional,
    Or,
    And,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    Equality,
    Relational,
    Shift,
    Additive,
    Multiplicative,
    Maximum,
}

impl Precedence {
    pub fn increment(self) -> Self {
        use Precedence::*;
        match self {
            Minimum => Assignment,
            Assignment => Conditional,
            Conditional => Or,
            Or => And,
            And => BitwiseOr,
            BitwiseOr => BitwiseXor,
            BitwiseXor => BitwiseAnd,
            BitwiseAnd => Equality,
            Equality => Relational,
            Relational => Shift,
            Shift => Additive,
            Additive => Multiplicative,
            Multiplicative => Maximum,
            Maximum => Maximum,
        }
    }
}

pub static BINARY_PRECEDENCES: LazyLock<HashMap<TokenType, Precedence>> = LazyLock::new(|| {
    [
        (Plus, Precedence::Additive),
        (PlusEqual, Precedence::Assignment),
        (Minus, Precedence::Additive),
        (MinusEqual, Precedence::Assignment),
        (Star, Precedence::Multiplicative),
        (StarEqual, Precedence::Assignment),
        (Slash, Precedence::Multiplicative),
        (SlashEqual, Precedence::Assignment),
        (Percent, Precedence::Multiplicative),
        (PercentEqual, Precedence::Assignment),
        (LessLess, Precedence::Shift),
        (LessLessEqual, Precedence::Assignment),
        (GreaterGreater, Precedence::Shift),
        (GreaterGreaterEqual, Precedence::Assignment),
        (Less, Precedence::Relational),
        (LessEqual, Precedence::Relational),
        (Greater, Precedence::Relational),
        (GreaterEqual, Precedence::Relational),
        (EqualEqual, Precedence::Equality),
        (BangEqual, Precedence::Equality),
        (And, Precedence::BitwiseAnd),
        (AndEqual, Precedence::Assignment),
        (Hat, Precedence::BitwiseXor),
        (HatEqual, Precedence::Assignment),
        (Pipe, Precedence::BitwiseOr),
        (PipeEqual, Precedence::Assignment),
        (AndAnd, Precedence::And),
        (PipePipe, Precedence::Or),
        (Question, Precedence::Conditional),
        (Equal, Precedence::Assignment),
    ]
    .into()
});

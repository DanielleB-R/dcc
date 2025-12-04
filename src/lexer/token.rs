use derive_more::Display;
use serde::{Deserialize, Serialize};

use crate::common::print_option;

#[derive(
    Clone, Copy, PartialEq, Eq, Debug, Hash, Display, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum TokenType {
    Identifier,
    IntConstant,
    LongConstant,
    UnsignedConstant,
    UnsignedLongConstant,
    FloatingPointConstant,
    CharConstant,
    StringLiteral,

    BreakKeyword,
    CaseKeyword,
    CharKeyword,
    ContinueKeyword,
    DefaultKeyword,
    DoKeyword,
    DoubleKeyword,
    ElseKeyword,
    ExternKeyword,
    ForKeyword,
    GotoKeyword,
    IfKeyword,
    IntKeyword,
    LongKeyword,
    ReturnKeyword,
    ShortKeyword,
    SignedKeyword,
    SizeofKeyword,
    StaticKeyword,
    StructKeyword,
    SwitchKeyword,
    UnsignedKeyword,
    VoidKeyword,
    WhileKeyword,

    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,

    Semicolon,

    And,
    AndAnd,
    AndEqual,
    Arrow,
    Bang,
    BangEqual,
    Colon,
    Comma,
    Dot,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    GreaterGreater,
    GreaterGreaterEqual,
    Hat,
    HatEqual,
    Less,
    LessEqual,
    LessLess,
    LessLessEqual,
    Minus,
    MinusEqual,
    MinusMinus,
    Percent,
    PercentEqual,
    Pipe,
    PipeEqual,
    PipePipe,
    Plus,
    PlusEqual,
    PlusPlus,
    Question,
    Slash,
    SlashEqual,
    Star,
    StarEqual,
    Tilde,

    EOF,
}

impl TokenType {
    pub fn is_type(&self) -> bool {
        use TokenType::*;
        matches!(
            self,
            IntKeyword
                | LongKeyword
                | UnsignedKeyword
                | SignedKeyword
                | DoubleKeyword
                | CharKeyword
                | VoidKeyword
                | StructKeyword
                | ShortKeyword
        )
    }

    pub fn is_type_qualifier(&self) -> bool {
        use TokenType::*;
        matches!(self, StaticKeyword | ExternKeyword)
    }

    pub fn is_specifier(&self) -> bool {
        self.is_type() || self.is_type_qualifier()
    }

    pub fn is_constant(&self) -> bool {
        self.is_integer_constant() || *self == Self::FloatingPointConstant
    }

    pub fn is_integer_constant(&self) -> bool {
        use TokenType::*;
        matches!(
            self,
            IntConstant | LongConstant | UnsignedConstant | UnsignedLongConstant | CharConstant
        )
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash, Display, Serialize, Deserialize)]
#[display(
    "{token_type:?} \"{}\" on line {line} at char {location}",
    print_option(value)
)]
pub struct Token {
    pub token_type: TokenType,
    pub value: Option<String>,
    #[serde(default)]
    pub location: usize,
    #[serde(default)]
    pub line: usize,
}

impl Token {
    pub fn new(token_type: TokenType, value: Option<String>, line: usize, location: usize) -> Self {
        Self {
            token_type,
            value,
            line,
            location,
        }
    }

    pub fn is_specifier(&self) -> bool {
        self.token_type.is_specifier()
    }
}

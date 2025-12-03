pub mod char_escape;
mod constant;
pub mod ctype;
pub mod type_table;

pub use constant::Constant;
pub use ctype::CType;

use std::{fmt::Display, hash::Hash};

use derive_more::Display;
use serde::Serialize;

use crate::lexer::token::Token;

#[derive(Clone, Copy, Debug, Serialize, Display)]
#[display("[{value} {line}:{location}]")]
pub struct Identifier {
    pub value: &'static str,
    pub line: usize,
    pub location: usize,
}

impl Identifier {
    pub fn relocate(mut self, other: &Identifier) -> Identifier {
        self.line = other.line;
        self.location = other.location;
        self
    }
}

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for Identifier {}

impl PartialOrd for Identifier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Identifier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(other.value)
    }
}

impl Hash for Identifier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

impl From<Token> for Identifier {
    fn from(value: Token) -> Self {
        Self {
            value: value.value.unwrap().leak(),
            line: value.line,
            location: value.location,
        }
    }
}

impl From<String> for Identifier {
    fn from(value: String) -> Self {
        Self {
            value: value.leak(),
            line: 0,
            location: 0,
        }
    }
}

impl From<&'static str> for Identifier {
    fn from(value: &'static str) -> Self {
        Self {
            value,
            line: 0,
            location: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Display, Hash)]
#[display("{tag}.{counter}")]
pub struct CodeLabel {
    pub tag: &'static str,
    pub counter: usize,
}

impl From<Token> for CodeLabel {
    fn from(value: Token) -> Self {
        Self {
            tag: value.value.unwrap().leak(),
            counter: 0,
        }
    }
}

pub fn print_option<T: Display>(option: &Option<T>) -> String {
    option
        .as_ref()
        .map(|inner| format!("{}", inner))
        .unwrap_or_else(|| "nil".to_owned())
}

pub fn print_vec<T: Display>(vector: &[T], separator: &str) -> String {
    vector
        .iter()
        .map(|param| format!("{}", param))
        .collect::<Vec<_>>()
        .join(separator)
}

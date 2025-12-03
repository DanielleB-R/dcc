use derive_more::{Display, From, IsVariant, Unwrap};
use serde::Serialize;
use std::collections::HashMap;

use crate::common::type_table::TypeTable;
use crate::common::{CType, Constant};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FunAttr {
    pub defined: bool,
    pub global: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct StaticAttr {
    pub init: InitialValue,
    pub global: bool,
}

#[derive(Debug, Clone, From, Unwrap, IsVariant, Serialize)]
#[unwrap(ref)]
pub enum IdentifierAttrs {
    Fun(FunAttr),
    Static(StaticAttr),
    Const(StaticInit),
    Local,
}

#[derive(Debug, Clone, Unwrap, PartialEq, Serialize, From)]
pub enum InitialValue {
    Tentative,
    Initial(Vec<StaticInit>),
    NoInitializer,
}

impl From<StaticInit> for InitialValue {
    fn from(value: StaticInit) -> Self {
        Self::Initial(vec![value])
    }
}

#[derive(Debug, Clone, PartialEq, Unwrap, Display, Serialize)]
pub enum StaticInit {
    #[display("(int {_0})")]
    IntInit(i32),
    #[display("(long {_0})")]
    LongInit(i64),
    #[display("(short {_0})")]
    ShortInit(i16),
    #[display("(unsigned {_0})")]
    UIntInit(u32),
    #[display("(unsigned long {_0})")]
    ULongInit(u64),
    #[display("(unsigned short {_0})")]
    UShortInit(u16),
    #[display("(double {_0})")]
    DoubleInit(f64),
    #[display("(char {_0})")]
    CharInit(i8),
    #[display("(unsigned char {_0})")]
    UCharInit(u8),
    #[display("(zeroes {_0})")]
    ZeroInit(usize),
    #[display("(str{} {_0})", if *_1 {"z"} else {""})]
    StringInit(String, bool),
    #[display("(ptr {_0})")]
    PointerInit(&'static str),
}

impl StaticInit {
    pub fn zero(value_type: &CType, type_table: &TypeTable) -> Self {
        match value_type {
            CType::Void => panic!(),
            CType::Int => Self::IntInit(0),
            CType::Long => Self::LongInit(0),
            CType::Short => Self::ShortInit(0),
            CType::Unsigned => Self::UIntInit(0),
            CType::UnsignedLong => Self::ULongInit(0),
            CType::UnsignedShort => Self::UShortInit(0),
            CType::Double => Self::DoubleInit(0.0),
            CType::Pointer(_) => Self::LongInit(0),
            CType::Array(_, _) => Self::ZeroInit(value_type.size(type_table)),
            CType::Char => Self::CharInit(0),
            CType::SignedChar => Self::CharInit(0),
            CType::UnsignedChar => Self::UCharInit(0),
            CType::Function(_) => panic!(),
            CType::Structure(tag) => {
                let struct_def = type_table
                    .get(&tag.value)
                    .expect("Struct should be defined");
                Self::ZeroInit(struct_def.size)
            }
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::IntInit(n) => *n == 0,
            Self::LongInit(n) => *n == 0,
            Self::ShortInit(n) => *n == 0,
            Self::UIntInit(n) => *n == 0,
            Self::ULongInit(n) => *n == 0,
            Self::UShortInit(n) => *n == 0,
            Self::DoubleInit(_) => false,
            Self::ZeroInit(_) => true,
            Self::CharInit(n) => *n == 0,
            Self::UCharInit(n) => *n == 0,
            Self::StringInit(_, _) => false,
            Self::PointerInit(_) => false,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::UShortInit(_) | Self::ShortInit(_) => 2,
            Self::IntInit(_) | Self::UIntInit(_) => 4,
            Self::CharInit(_) | Self::UCharInit(_) => 1,
            Self::ZeroInit(n) => *n,
            _ => 8,
            // this is wrong
        }
    }

    pub fn from_constant(c: Constant, value_type: &CType) -> Self {
        match c.convert_type(value_type) {
            Constant::Int(n) => Self::IntInit(n),
            Constant::Long(n) => Self::LongInit(n),
            Constant::Short(n) => Self::ShortInit(n),
            Constant::UInt(n) => Self::UIntInit(n),
            Constant::ULong(n) => Self::ULongInit(n),
            Constant::UShort(n) => Self::UShortInit(n),
            Constant::Double(n) => Self::DoubleInit(n),
            Constant::Char(n) => Self::CharInit(n),
            Constant::UChar(n) => Self::UCharInit(n),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SymbolEntry {
    pub c_type: CType,
    pub attrs: IdentifierAttrs,
}

pub type SymbolTable = HashMap<&'static str, SymbolEntry>;

pub trait STable {
    fn get_expected_type(&self, key: &str) -> CType;
}

impl STable for SymbolTable {
    fn get_expected_type(&self, key: &str) -> CType {
        self.get(key)
            .expect("Name should be resolved")
            .c_type
            .clone()
    }
}

use derive_more::{Display, IsVariant, Unwrap};
use serde::Serialize;

use super::{
    Identifier, print_vec,
    type_table::{TTable, TypeTable},
};

#[derive(Debug, Clone, PartialEq, Eq, Display, Serialize, IsVariant, Unwrap)]
#[unwrap(ref)]
pub enum CType {
    Void,
    Int,
    Long,
    Short,
    Unsigned,
    UnsignedLong,
    UnsignedShort,
    Char,
    SignedChar,
    UnsignedChar,
    Double,
    #[display("Pointer {_0}")]
    Pointer(Box<CType>),
    #[display("Array({_0}; {_1})")]
    Array(Box<CType>, usize),
    #[display("Function({}; {})", print_vec(&_0.params, ", "), _0.ret)]
    Function(Box<FunctionType>),
    #[display("Struct {_0}")]
    Structure(Identifier),
}

impl From<FunctionType> for CType {
    fn from(value: FunctionType) -> Self {
        Self::Function(Box::new(value))
    }
}

impl CType {
    pub fn for_string(s: &str) -> Self {
        Self::Array(Box::new(Self::Char), s.len() + 1)
    }

    pub fn pointer_to(base_type: Self) -> Self {
        Self::Pointer(Box::from(base_type))
    }

    pub fn array_of(base_type: Self, size: usize) -> Self {
        Self::Array(Box::from(base_type), size)
    }

    pub fn size(&self, type_table: &TypeTable) -> usize {
        use CType::*;
        match self {
            Void => panic!("Void is unsized"),
            Int => 4,
            Long => 8,
            Short => 2,
            UnsignedShort => 2,
            Unsigned => 4,
            UnsignedLong => 8,
            Char => 1,
            SignedChar => 1,
            UnsignedChar => 1,
            Double => 8,
            Pointer(_) => 8,
            Array(element_type, size) => element_type.size(type_table) * size,
            Function(_) => panic!("Functions are unsized"),
            Structure(tag) => {
                type_table
                    .get(&tag.value)
                    .expect("by now it should be complete")
                    .size
            }
        }
    }

    pub fn returns_in_memory(&self, type_table: &TypeTable) -> bool {
        match self {
            Self::Structure(tag) => {
                type_table
                    .get(&tag.value)
                    .map(|e| e.size)
                    .unwrap_or_default()
                    > 16
            }
            _ => false,
        }
    }

    pub fn referent_size(&self, type_table: &TypeTable) -> usize {
        self.unwrap_pointer_ref().size(type_table)
    }

    pub fn scalar_size(&self, type_table: &TypeTable) -> usize {
        use CType::*;
        match self {
            Void => panic!("Void is unsized"),
            Int => 4,
            Long => 8,
            Short => 2,
            UnsignedShort => 2,
            Unsigned => 4,
            UnsignedLong => 8,
            Char => 1,
            SignedChar => 1,
            UnsignedChar => 1,
            Double => 8,
            Pointer(_) => 8,
            Array(element_type, _) => element_type.scalar_size(type_table),
            Function(_) => panic!("Functions are unsized"),
            Structure(tag) => type_table.get_expected(tag).size,
        }
    }

    pub fn alignment(&self, type_table: &TypeTable) -> usize {
        use CType::*;
        match self {
            Void => panic!("Void does not have an alignment"),
            Short | UnsignedShort => 2,
            Int => 4,
            Long => 8,
            Unsigned => 4,
            UnsignedLong => 8,
            Double => 8,
            Pointer(_) => 8,
            Char | SignedChar | UnsignedChar => 1,
            Array(value_type, size) => {
                if value_type.size(type_table) * size < 16 {
                    value_type.scalar_size(type_table)
                } else {
                    16
                }
            }
            Function(_) => panic!("Functions do not have an alignment"),
            Structure(tag) => {
                type_table
                    .get(&tag.value)
                    .expect("we only get alignment of complete structs")
                    .alignment
            }
        }
    }

    pub fn type_alignment(&self, type_table: &TypeTable) -> usize {
        use CType::*;
        match self {
            Void => panic!("Void does not have an alignment"),
            Short | UnsignedShort => 2,
            Int => 4,
            Long => 8,
            Unsigned => 4,
            UnsignedLong => 8,
            Double => 8,
            Pointer(_) => 8,
            Char | SignedChar | UnsignedChar => 1,
            Array(value_type, _) => value_type.alignment(type_table),
            Function(_) => panic!("Functions do not have an alignment"),
            Structure(tag) => {
                type_table
                    .get(&tag.value)
                    .expect("we only get alignment of complete structs")
                    .alignment
            }
        }
    }

    pub fn is_signed(&self) -> bool {
        use CType::*;
        matches!(self, Int | Long | Short | Char | SignedChar)
    }

    pub fn is_character(&self) -> bool {
        use CType::*;
        matches!(self, Char | SignedChar | UnsignedChar)
    }

    pub fn is_arithmetic(&self) -> bool {
        use CType::*;
        matches!(
            self,
            Int | Long
                | Short
                | Unsigned
                | UnsignedLong
                | UnsignedShort
                | Double
                | Char
                | SignedChar
                | UnsignedChar
        )
    }

    pub fn is_integer(&self) -> bool {
        use CType::*;
        matches!(
            self,
            Int | Long
                | Short
                | Unsigned
                | UnsignedLong
                | UnsignedShort
                | Char
                | SignedChar
                | UnsignedChar
        )
    }

    pub fn is_scalar(&self) -> bool {
        !matches!(self, Self::Void | Self::Array(..) | Self::Structure(_))
    }

    pub fn is_void_pointer(&self) -> bool {
        match self {
            Self::Pointer(inner_type) => **inner_type == Self::Void,
            _ => false,
        }
    }

    pub fn is_complete(&self, type_table: &TypeTable) -> bool {
        match self {
            Self::Void => false,
            Self::Structure(tag) => {
                type_table.contains_key(&tag.value)
                    && !type_table.get(&tag.value).unwrap().members.is_empty()
            }
            _ => true,
        }
    }

    pub fn is_pointer_to_complete(&self, type_table: &TypeTable) -> bool {
        match self {
            Self::Pointer(inner_type) => inner_type.is_complete(type_table),
            _ => false,
        }
    }

    pub fn is_valid_struct_member(&self, type_table: &TypeTable) -> bool {
        if !self.is_complete(type_table) {
            return false;
        }
        match self {
            Self::Array(element, _) => element.is_valid_struct_member(type_table),
            Self::Pointer(referent) => {
                if referent.is_array() {
                    referent.is_valid_struct_member(type_table)
                } else {
                    true
                }
            }
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FunctionType {
    pub params: Vec<CType>,
    pub ret: CType,
}

use derive_more::{Display, From, IsVariant, Unwrap};
use serde::Serialize;
use std::{
    hash::Hash,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub},
};

use super::{CType, type_table::TypeTable};

#[derive(Debug, Clone, Copy, Display, From, Unwrap, Serialize, PartialOrd, IsVariant)]
pub enum Constant {
    Int(i32),
    Long(i64),
    Short(i16),
    UInt(u32),
    ULong(u64),
    UShort(u16),
    Double(f64),
    Char(i8),
    UChar(u8),
}

impl Constant {
    pub const ZERO: Self = Self::Int(0);
    pub const ONE: Self = Self::Int(1);

    pub fn one(value_type: &CType, type_table: &TypeTable) -> Self {
        use Constant::*;
        match value_type {
            CType::Void => panic!("no one value for void"),
            CType::Int => Int(1),
            CType::Long => Long(1),
            CType::Unsigned => UInt(1),
            CType::UnsignedLong => ULong(1),
            CType::Short => Short(1),
            CType::UnsignedShort => UShort(1),
            CType::Double => Double(1.0),
            CType::Pointer(inner_type) => Long(inner_type.size(type_table) as i64),
            CType::Char => Char(1),
            CType::SignedChar => Char(1),
            CType::UnsignedChar => UChar(1),
            CType::Array(_, _) => panic!("no one value for arrays"),
            CType::Function(_) => panic!("no one value for functions"),
            CType::Structure(_) => panic!("no one value for structures"),
        }
    }

    pub fn is_zero(&self) -> bool {
        use Constant::*;
        matches!(
            self,
            Int(0)
                | Long(0)
                | Short(0)
                | UInt(0)
                | ULong(0)
                | UShort(0)
                | Char(0)
                | UChar(0)
                | Double(0.0)
        )
    }

    pub fn logical_not(&self) -> Self {
        if self.is_zero() {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    pub fn size(&self) -> usize {
        use Constant::*;
        match self {
            Int(_) => 4,
            Long(_) => 8,
            UInt(_) => 4,
            ULong(_) => 8,
            Short(_) => 2,
            UShort(_) => 2,
            Double(_) => 8,
            Char(_) => 1,
            UChar(_) => 1,
        }
    }

    pub fn get_type(&self) -> CType {
        use Constant::*;
        match self {
            Int(_) => CType::Int,
            Long(_) => CType::Long,
            UInt(_) => CType::Unsigned,
            ULong(_) => CType::UnsignedLong,
            Short(_) => CType::Short,
            UShort(_) => CType::UnsignedShort,
            Double(_) => CType::Double,
            Char(_) => CType::Char,
            UChar(_) => CType::UnsignedChar,
        }
    }

    pub fn is_null_pointer_constant(&self) -> bool {
        use Constant::*;
        matches!(self, Int(0) | Long(0) | UInt(0) | ULong(0))
    }

    pub fn unwrap_integer(self) -> i64 {
        use Constant::*;
        match self {
            Int(n) => n as i64,
            Long(n) => n,
            Short(n) => n as i64,
            UInt(n) => n as i64,
            ULong(n) => n as i64,
            UShort(n) => n as i64,
            Double(_) => panic!("No integer value of double"),
            Char(n) => n as i64,
            UChar(n) => n as i64,
        }
    }

    pub fn convert_type(self, value_type: &CType) -> Self {
        use Constant::*;
        match (self, value_type) {
            (Int(n), CType::Int) => Int(n),
            (Int(n), CType::Long) => Long(n as i64),
            (Int(n), CType::Unsigned) => UInt(n as u32),
            (Int(n), CType::UnsignedLong) => ULong(n as u64),
            (Int(n), CType::Short) => Short(n as i16),
            (Int(n), CType::UnsignedShort) => UShort(n as u16),
            (Int(n), CType::Double) => Double(n as f64),
            (Int(n), CType::Char) => Char(n as i8),
            (Int(n), CType::SignedChar) => Char(n as i8),
            (Int(n), CType::UnsignedChar) => UChar(n as u8),
            (Int(n), CType::Pointer(_)) => ULong(n as u64),
            (Long(n), CType::Int) => Int(n as i32),
            (Long(n), CType::Long) => Long(n),
            (Long(n), CType::Unsigned) => UInt(n as u32),
            (Long(n), CType::UnsignedLong) => ULong(n as u64),
            (Long(n), CType::Short) => Short(n as i16),
            (Long(n), CType::UnsignedShort) => UShort(n as u16),
            (Long(n), CType::Double) => Double(n as f64),
            (Long(n), CType::Char) => Char(n as i8),
            (Long(n), CType::SignedChar) => Char(n as i8),
            (Long(n), CType::UnsignedChar) => UChar(n as u8),
            (Long(n), CType::Pointer(_)) => ULong(n as u64),
            (UInt(n), CType::Int) => Int(n as i32),
            (UInt(n), CType::Long) => Long(n as i64),
            (UInt(n), CType::Unsigned) => UInt(n),
            (UInt(n), CType::UnsignedLong) => ULong(n as u64),
            (UInt(n), CType::Short) => Short(n as i16),
            (UInt(n), CType::UnsignedShort) => UShort(n as u16),
            (UInt(n), CType::Double) => Double(n as f64),
            (UInt(n), CType::Char) => Char(n as i8),
            (UInt(n), CType::SignedChar) => Char(n as i8),
            (UInt(n), CType::UnsignedChar) => UChar(n as u8),
            (UInt(n), CType::Pointer(_)) => ULong(n as u64),
            (ULong(n), CType::Int) => Int(n as i32),
            (ULong(n), CType::Long) => Long(n as i64),
            (ULong(n), CType::Unsigned) => UInt(n as u32),
            (ULong(n), CType::UnsignedLong) => ULong(n),
            (ULong(n), CType::Short) => Short(n as i16),
            (ULong(n), CType::UnsignedShort) => UShort(n as u16),
            (ULong(n), CType::Double) => Double(n as f64),
            (ULong(n), CType::Char) => Char(n as i8),
            (ULong(n), CType::SignedChar) => Char(n as i8),
            (ULong(n), CType::UnsignedChar) => UChar(n as u8),
            (ULong(n), CType::Pointer(_)) => ULong(n),
            (Short(n), CType::Int) => Int(n as i32),
            (Short(n), CType::Long) => Long(n as i64),
            (Short(n), CType::Unsigned) => UInt(n as u32),
            (Short(n), CType::UnsignedLong) => ULong(n as u64),
            (Short(n), CType::Short) => Short(n),
            (Short(n), CType::UnsignedShort) => UShort(n as u16),
            (Short(n), CType::Double) => Double(n as f64),
            (Short(n), CType::Char) => Char(n as i8),
            (Short(n), CType::SignedChar) => Char(n as i8),
            (Short(n), CType::UnsignedChar) => UChar(n as u8),
            (Short(n), CType::Pointer(_)) => ULong(n as u64),
            (UShort(n), CType::Int) => Int(n as i32),
            (UShort(n), CType::Long) => Long(n as i64),
            (UShort(n), CType::Unsigned) => UInt(n as u32),
            (UShort(n), CType::UnsignedLong) => ULong(n as u64),
            (UShort(n), CType::Short) => Short(n as i16),
            (UShort(n), CType::UnsignedShort) => UShort(n),
            (UShort(n), CType::Double) => Double(n as f64),
            (UShort(n), CType::Char) => Char(n as i8),
            (UShort(n), CType::SignedChar) => Char(n as i8),
            (UShort(n), CType::UnsignedChar) => UChar(n as u8),
            (UShort(n), CType::Pointer(_)) => ULong(n as u64),
            (Double(n), CType::Int) => Int(n as i32),
            (Double(n), CType::Long) => Long(n as i64),
            (Double(n), CType::Unsigned) => UInt(n as u32),
            (Double(n), CType::UnsignedLong) => ULong(n as u64),
            (Double(n), CType::Short) => Short(n as i16),
            (Double(n), CType::UnsignedShort) => UShort(n as u16),
            (Double(n), CType::Double) => Double(n),
            (Double(n), CType::Char) => Char(n as i8),
            (Double(n), CType::SignedChar) => Char(n as i8),
            (Double(n), CType::UnsignedChar) => UChar(n as u8),
            (Double(_), CType::Pointer(_)) => panic!(),
            (Char(n), CType::Int) => Int(n as i32),
            (Char(n), CType::Long) => Long(n as i64),
            (Char(n), CType::Unsigned) => UInt(n as u32),
            (Char(n), CType::UnsignedLong) => ULong(n as u64),
            (Char(n), CType::Short) => Short(n as i16),
            (Char(n), CType::UnsignedShort) => UShort(n as u16),
            (Char(n), CType::Double) => Double(n as f64),
            (Char(n), CType::Char) => Char(n),
            (Char(n), CType::SignedChar) => Char(n),
            (Char(n), CType::UnsignedChar) => UChar(n as u8),
            (Char(n), CType::Pointer(_)) => ULong(n as u64),
            (UChar(n), CType::Int) => Int(n as i32),
            (UChar(n), CType::Long) => Long(n as i64),
            (UChar(n), CType::Unsigned) => UInt(n as u32),
            (UChar(n), CType::UnsignedLong) => ULong(n as u64),
            (UChar(n), CType::Short) => Short(n as i16),
            (UChar(n), CType::UnsignedShort) => UShort(n as u16),
            (UChar(n), CType::Double) => Double(n as f64),
            (UChar(n), CType::Char) => Char(n as i8),
            (UChar(n), CType::SignedChar) => Char(n as i8),
            (UChar(n), CType::UnsignedChar) => UChar(n),
            (UChar(n), CType::Pointer(_)) => ULong(n as u64),
            // just return a placeholder
            (_, CType::Void) => Int(0),
            pair => {
                eprintln!("{:?}", pair);
                unimplemented!();
            }
        }
    }

    pub fn c_equals(&self, other: &Self) -> bool {
        use Constant::*;
        match (self, other) {
            (Int(n1), Int(n2)) => n1 == n2,
            (Long(n1), Long(n2)) => n1 == n2,
            (Short(n1), Short(n2)) => n1 == n2,
            (UInt(n1), UInt(n2)) => n1 == n2,
            (ULong(n1), ULong(n2)) => n1 == n2,
            (UShort(n1), UShort(n2)) => n1 == n2,
            (Char(n1), Char(n2)) => n1 == n2,
            (UChar(n1), UChar(n2)) => n1 == n2,
            (Double(f1), Double(f2)) => f1 == f2,
            _ => false,
        }
    }
}

impl Add for Constant {
    type Output = Constant;

    fn add(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a.wrapping_add(b)),
            (Long(a), Long(b)) => Long(a.wrapping_add(b)),
            (Short(a), Short(b)) => Short(a.wrapping_add(b)),
            (UInt(a), UInt(b)) => UInt(a.wrapping_add(b)),
            (ULong(a), ULong(b)) => ULong(a.wrapping_add(b)),
            (UShort(a), UShort(b)) => UShort(a.wrapping_add(b)),
            (Char(a), Char(b)) => Char(a.wrapping_add(b)),
            (UChar(a), UChar(b)) => UChar(a.wrapping_add(b)),
            (Double(a), Double(b)) => Double(a + b),
            _ => panic!(),
        }
    }
}

impl Sub for Constant {
    type Output = Constant;

    fn sub(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a.wrapping_sub(b)),
            (Long(a), Long(b)) => Long(a.wrapping_sub(b)),
            (Short(a), Short(b)) => Short(a.wrapping_sub(b)),
            (UInt(a), UInt(b)) => UInt(a.wrapping_sub(b)),
            (ULong(a), ULong(b)) => ULong(a.wrapping_sub(b)),
            (UShort(a), UShort(b)) => UShort(a.wrapping_sub(b)),
            (Char(a), Char(b)) => Char(a.wrapping_sub(b)),
            (UChar(a), UChar(b)) => UChar(a.wrapping_sub(b)),
            (Double(a), Double(b)) => Double(a - b),
            _ => panic!(),
        }
    }
}

impl Mul for Constant {
    type Output = Constant;

    fn mul(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a.wrapping_mul(b)),
            (Long(a), Long(b)) => Long(a.wrapping_mul(b)),
            (Short(a), Short(b)) => Short(a.wrapping_mul(b)),
            (UInt(a), UInt(b)) => UInt(a.wrapping_mul(b)),
            (ULong(a), ULong(b)) => ULong(a.wrapping_mul(b)),
            (UShort(a), UShort(b)) => UShort(a.wrapping_mul(b)),
            (Char(a), Char(b)) => Char(a.wrapping_mul(b)),
            (UChar(a), UChar(b)) => UChar(a.wrapping_mul(b)),
            (Double(a), Double(b)) => Double(a * b),
            _ => panic!(),
        }
    }
}

impl Div for Constant {
    type Output = Constant;

    fn div(self, rhs: Self) -> Self::Output {
        use Constant::*;
        if !rhs.is_double() && rhs.is_zero() {
            Int(0)
        } else {
            match (self, rhs) {
                (Int(a), Int(b)) => Int(a.wrapping_div(b)),
                (Long(a), Long(b)) => Long(a.wrapping_div(b)),
                (Short(a), Short(b)) => Short(a.wrapping_div(b)),
                (UInt(a), UInt(b)) => UInt(a.wrapping_div(b)),
                (ULong(a), ULong(b)) => ULong(a.wrapping_div(b)),
                (UShort(a), UShort(b)) => UShort(a.wrapping_div(b)),
                (Char(a), Char(b)) => Char(a.wrapping_div(b)),
                (UChar(a), UChar(b)) => UChar(a.wrapping_div(b)),
                (Double(a), Double(b)) => Double(a / b),
                _ => panic!(),
            }
        }
    }
}

impl Rem for Constant {
    type Output = Constant;

    fn rem(self, rhs: Self) -> Self::Output {
        use Constant::*;
        if rhs.is_zero() {
            Int(0)
        } else {
            match (self, rhs) {
                (Int(a), Int(b)) => Int(a.wrapping_rem(b)),
                (Long(a), Long(b)) => Long(a.wrapping_rem(b)),
                (Short(a), Short(b)) => Short(a.wrapping_rem(b)),
                (UInt(a), UInt(b)) => UInt(a.wrapping_rem(b)),
                (ULong(a), ULong(b)) => ULong(a.wrapping_rem(b)),
                (UShort(a), UShort(b)) => UShort(a.wrapping_rem(b)),
                (Char(a), Char(b)) => Char(a.wrapping_rem(b)),
                (UChar(a), UChar(b)) => UChar(a.wrapping_rem(b)),
                _ => panic!(),
            }
        }
    }
}

impl BitAnd for Constant {
    type Output = Constant;

    fn bitand(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a & b),
            (Long(a), Long(b)) => Long(a & b),
            (Short(a), Short(b)) => Short(a & b),
            (UInt(a), UInt(b)) => UInt(a & b),
            (ULong(a), ULong(b)) => ULong(a & b),
            (UShort(a), UShort(b)) => UShort(a & b),
            (Char(a), Char(b)) => Char(a & b),
            (UChar(a), UChar(b)) => UChar(a & b),
            _ => panic!(),
        }
    }
}

impl BitOr for Constant {
    type Output = Constant;

    fn bitor(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a | b),
            (Long(a), Long(b)) => Long(a | b),
            (Short(a), Short(b)) => Short(a | b),
            (UInt(a), UInt(b)) => UInt(a | b),
            (ULong(a), ULong(b)) => ULong(a | b),
            (UShort(a), UShort(b)) => UShort(a | b),
            (Char(a), Char(b)) => Char(a | b),
            (UChar(a), UChar(b)) => UChar(a | b),
            _ => panic!(),
        }
    }
}

impl BitXor for Constant {
    type Output = Constant;

    fn bitxor(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a ^ b),
            (Long(a), Long(b)) => Long(a ^ b),
            (Short(a), Short(b)) => Short(a ^ b),
            (UInt(a), UInt(b)) => UInt(a ^ b),
            (ULong(a), ULong(b)) => ULong(a ^ b),
            (UShort(a), UShort(b)) => UShort(a ^ b),
            (Char(a), Char(b)) => Char(a ^ b),
            (UChar(a), UChar(b)) => UChar(a ^ b),
            _ => panic!(),
        }
    }
}

impl Shl for Constant {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a.unbounded_shl(b as u32)),
            (Long(a), Long(b)) => Long(a.unbounded_shl(b as u32)),
            (Short(a), Short(b)) => Short(a.unbounded_shl(b as u32)),
            (UInt(a), UInt(b)) => UInt(a.unbounded_shl(b)),
            (ULong(a), ULong(b)) => ULong(a.unbounded_shl(b as u32)),
            (UShort(a), UShort(b)) => UShort(a.unbounded_shl(b as u32)),
            (Char(a), Char(b)) => Char(a.unbounded_shl(b as u32)),
            (UChar(a), UChar(b)) => UChar(a.unbounded_shl(b as u32)),
            _ => panic!(),
        }
    }
}

impl Shr for Constant {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        use Constant::*;
        match (self, rhs) {
            (Int(a), Int(b)) => Int(a.unbounded_shr(b as u32)),
            (Long(a), Long(b)) => Long(a.unbounded_shr(b as u32)),
            (Short(a), Short(b)) => Short(a.unbounded_shr(b as u32)),
            (UInt(a), UInt(b)) => UInt(a.unbounded_shr(b)),
            (ULong(a), ULong(b)) => ULong(a.unbounded_shr(b as u32)),
            (UShort(a), UShort(b)) => UShort(a.unbounded_shr(b as u32)),
            (Char(a), Char(b)) => Char(a.unbounded_shr(b as u32)),
            (UChar(a), UChar(b)) => UChar(a.unbounded_shr(b as u32)),
            _ => panic!(),
        }
    }
}

impl Neg for Constant {
    type Output = Self;

    fn neg(self) -> Self::Output {
        use Constant::*;
        match self {
            Int(n) => Int(-n),
            Long(n) => Long(-n),
            Short(n) => Short(-n),
            UInt(n) => UInt(n.wrapping_neg()),
            ULong(n) => ULong(n.wrapping_neg()),
            UShort(n) => UShort(n.wrapping_neg()),
            Double(n) => Double(-n),
            Char(n) => Char(-n),
            UChar(n) => UChar(n.wrapping_neg()),
        }
    }
}

impl Not for Constant {
    type Output = Self;

    fn not(self) -> Self::Output {
        use Constant::*;
        match self {
            Int(n) => Int(!n),
            Long(n) => Long(!n),
            Short(n) => Short(!n),
            UInt(n) => UInt(!n),
            ULong(n) => ULong(!n),
            UShort(n) => UShort(!n),
            Char(n) => Char(!n),
            UChar(n) => UChar(!n),
            _ => panic!(),
        }
    }
}

impl From<bool> for Constant {
    fn from(value: bool) -> Self {
        if value { Self::Int(1) } else { Self::Int(0) }
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        use Constant::*;
        match (self, other) {
            (Int(n1), Int(n2)) => n1 == n2,
            (Long(n1), Long(n2)) => n1 == n2,
            (Short(n1), Short(n2)) => n1 == n2,
            (UInt(n1), UInt(n2)) => n1 == n2,
            (ULong(n1), ULong(n2)) => n1 == n2,
            (UShort(n1), UShort(n2)) => n1 == n2,
            (Char(n1), Char(n2)) => n1 == n2,
            (UChar(n1), UChar(n2)) => n1 == n2,
            (Double(f1), Double(f2)) => f1.to_bits() == f2.to_bits(),
            _ => false,
        }
    }
}

impl Eq for Constant {}

impl Hash for Constant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use Constant::*;
        match self {
            Int(n) => {
                state.write_u8(1);
                n.hash(state);
            }
            Long(n) => {
                state.write_u8(2);
                n.hash(state);
            }
            UInt(n) => {
                state.write_u8(3);
                n.hash(state);
            }
            ULong(n) => {
                state.write_u8(4);
                n.hash(state);
            }
            Double(f) => {
                state.write_u8(5);
                f.to_bits().hash(state);
            }
            Char(n) => {
                state.write_u8(6);
                n.hash(state);
            }
            UChar(n) => {
                state.write_u8(7);
                n.hash(state);
            }
            Short(n) => {
                state.write_u8(8);
                n.hash(state);
            }
            UShort(n) => {
                state.write_u8(9);
                n.hash(state);
            }
        }
    }
}

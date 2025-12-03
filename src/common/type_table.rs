use std::collections::HashMap;

use serde::Serialize;

use crate::common::{CType, Identifier};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EightbyteClass {
    MEMORY,
    SSE,
    INTEGER,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct StructEntry {
    pub alignment: usize,
    pub size: usize,
    pub members: Vec<MemberEntry>,
}

impl StructEntry {
    pub fn find_offset(&self, member_name: Identifier) -> usize {
        self.members
            .iter()
            .find(|m| m.name.value == member_name.value)
            .unwrap()
            .offset
    }

    pub fn classify(&self, types: &TypeTable) -> Vec<EightbyteClass> {
        use EightbyteClass::*;
        if self.size > 16 {
            return vec![MEMORY; self.size / 8 + if self.size.is_multiple_of(8) { 0 } else { 1 }];
        }

        let scalars = flatten_members(&self.members, types);
        if self.size > 8 {
            match scalars[..] {
                [CType::Double, CType::Double] => vec![SSE, SSE],
                [CType::Double, ..] => vec![SSE, INTEGER],
                [.., CType::Double] => vec![INTEGER, SSE],
                _ => vec![INTEGER, INTEGER],
            }
        } else if scalars[0] == CType::Double {
            vec![SSE]
        } else {
            vec![INTEGER]
        }
    }
}

fn flatten_members(members: &[MemberEntry], types: &TypeTable) -> Vec<CType> {
    members
        .iter()
        .flat_map(|m| match &m.member_type {
            CType::Array(element, size) => vec![*element.clone(); *size],
            CType::Structure(tag) => flatten_members(&types.get_expected(tag).members, types),
            t => vec![t.clone()],
        })
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct MemberEntry {
    pub name: Identifier,
    pub member_type: CType,
    pub offset: usize,
}

pub type TypeTable = HashMap<&'static str, StructEntry>;

pub trait TTable {
    fn get_expected(&self, key: &Identifier) -> &StructEntry;
}

impl TTable for TypeTable {
    fn get_expected(&self, key: &Identifier) -> &StructEntry {
        self.get(&key.value).unwrap()
    }
}

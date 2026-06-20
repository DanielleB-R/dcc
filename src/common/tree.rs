use derive_more::{Display, From};
use serde::Serialize;

use super::print_vec;

#[derive(Clone, Debug, Display, From, Serialize)]
#[display("{}", print_vec(declarations, "\n\n"))]
pub struct Program<T: std::fmt::Display> {
    pub declarations: Vec<T>,
}

impl<T: std::fmt::Display> Program<T> {
    pub fn map<E, F: FnMut(T) -> Result<T, E>>(self, transform: F) -> Result<Self, E> {
        Ok(self
            .declarations
            .into_iter()
            .map(transform)
            .collect::<Result<Vec<_>, E>>()?
            .into())
    }
}

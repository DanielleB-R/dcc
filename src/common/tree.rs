use derive_more::{Display, From};
use serde::Serialize;

use super::print_vec;

#[derive(Clone, Debug, Display, From, Serialize)]
#[display("{}", print_vec(top_level, "\n\n"))]
pub struct Program<T: std::fmt::Display> {
    pub top_level: Vec<T>,
}

impl<T: std::fmt::Display> Program<T> {
    pub fn map<E, F: FnMut(T) -> Result<T, E>>(self, transform: F) -> Result<Self, E> {
        Ok(self
            .top_level
            .into_iter()
            .map(transform)
            .collect::<Result<Vec<_>, E>>()?
            .into())
    }

    pub fn map_infallible<F: FnMut(T) -> T>(self, transform: F) -> Self {
        self.top_level
            .into_iter()
            .map(transform)
            .collect::<Vec<_>>()
            .into()
    }
}

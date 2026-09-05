use crate::{
    common::{symbol_table::SymbolTable, type_table::TypeTable},
    errors::CompilerError,
    tacky::ir,
};

pub(crate) trait Backend {
    fn emit(
        self,
        code: ir::Program,
        symbols: SymbolTable,
        types: &TypeTable,
    ) -> Result<String, CompilerError>;
}

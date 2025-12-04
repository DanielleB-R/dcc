use crate::common::CodeLabel;

pub fn emit_label(label: &str) -> String {
    #[cfg(target_os = "macos")]
    return format!("_{}", label);
    #[cfg(target_os = "linux")]
    return label.to_owned();
}

pub fn emit_local_label(label: CodeLabel) -> String {
    #[cfg(target_os = "macos")]
    return format!("L{}", label);

    #[cfg(target_os = "linux")]
    return format!(".L{}", label);
}

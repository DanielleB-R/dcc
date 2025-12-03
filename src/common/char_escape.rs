use std::{
    collections::{HashMap, VecDeque},
    sync::LazyLock,
};

static ESCAPE_SEQUENCES: LazyLock<HashMap<char, char>> = LazyLock::new(|| {
    [
        ('\'', '\''),
        ('"', '"'),
        ('?', '?'),
        ('\\', '\\'),
        ('a', 7 as char),
        ('b', 8 as char),
        ('f', 12 as char),
        ('n', 10 as char),
        ('r', 13 as char),
        ('t', 9 as char),
        ('v', 11 as char),
    ]
    .into()
});

pub fn unescape(input: &str) -> String {
    let mut char_deque: VecDeque<_> = input.chars().collect();

    let mut result = String::new();

    while let Some(next_char) = char_deque.pop_front() {
        if next_char == '\\' {
            result.push(
                ESCAPE_SEQUENCES
                    .get(&char_deque.pop_front().unwrap())
                    .copied()
                    .unwrap(),
            )
        } else {
            result.push(next_char);
        }
    }

    result
}

pub fn escape(s: &str) -> String {
    let mut result = String::new();

    for c in s.chars() {
        if c.is_ascii_control() {
            result.push_str(&format!("\\{:o}", c as u32));
        } else if c == '"' {
            result.push_str("\\\"");
        } else if c == '\\' {
            result.push_str("\\\\");
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_string_escaping() {
        assert_eq!(escape("\u{7}\u{8}"), "\\7\\10");
    }
}

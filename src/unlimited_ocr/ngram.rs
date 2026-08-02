use std::collections::BTreeSet;

use a3s_use_core::{UseError, UseResult};

pub(crate) const NO_REPEAT_NGRAM_SIZE: usize = 35;
pub(crate) const NO_REPEAT_WINDOW: usize = 128;

#[derive(Debug, Clone)]
pub(crate) struct SlidingNoRepeatNgram {
    size: usize,
    window: usize,
    whitelist: BTreeSet<u32>,
}

impl SlidingNoRepeatNgram {
    pub(crate) fn reviewed() -> Self {
        Self {
            size: NO_REPEAT_NGRAM_SIZE,
            window: NO_REPEAT_WINDOW,
            whitelist: BTreeSet::new(),
        }
    }

    #[cfg(test)]
    fn new(size: usize, window: usize, whitelist: impl IntoIterator<Item = u32>) -> Self {
        Self {
            size,
            window,
            whitelist: whitelist.into_iter().collect(),
        }
    }

    pub(crate) fn ban_in_place(&self, sequence: &[u32], logits: &mut [f32]) -> UseResult<()> {
        let vocabulary = logits.len();
        for token in self.banned_tokens(sequence) {
            let index = usize::try_from(token).map_err(|_| {
                generation_error("Unlimited-OCR generated a token outside the host index range.")
            })?;
            let logit = logits.get_mut(index).ok_or_else(|| {
                generation_error(format!(
                    "Unlimited-OCR no-repeat state referenced token {token}, outside the {}-token vocabulary.",
                    vocabulary
                ))
            })?;
            *logit = f32::NEG_INFINITY;
        }
        Ok(())
    }

    fn banned_tokens(&self, sequence: &[u32]) -> BTreeSet<u32> {
        if self.size == 0 || self.window == 0 || sequence.len() < self.size {
            return BTreeSet::new();
        }
        let search_start = sequence.len().saturating_sub(self.window);
        let search_end = sequence.len() - self.size + 1;
        if search_end <= search_start {
            return BTreeSet::new();
        }
        let prefix = if self.size > 1 {
            &sequence[sequence.len() - (self.size - 1)..]
        } else {
            &[]
        };
        let mut banned = BTreeSet::new();
        for start in search_start..search_end {
            let ngram = &sequence[start..start + self.size];
            if self.size == 1 || &ngram[..self.size - 1] == prefix {
                banned.insert(ngram[self.size - 1]);
            }
        }
        banned.retain(|token| !self.whitelist.contains(token));
        banned
    }
}

pub(crate) fn greedy_token(logits: &[f32]) -> UseResult<u32> {
    let (index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .ok_or_else(|| generation_error("Unlimited-OCR produced no finite token logits."))?;
    u32::try_from(index)
        .map_err(|_| generation_error("Unlimited-OCR vocabulary index exceeds u32."))
}

fn generation_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.generation_failed", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_are_banned_only_inside_the_trailing_window() {
        let processor = SlidingNoRepeatNgram::new(3, 8, []);
        assert_eq!(
            processor.banned_tokens(&[5, 6, 7, 6, 7]),
            BTreeSet::from([6])
        );
        assert!(processor.banned_tokens(&[1, 2]).is_empty());
    }

    #[test]
    fn whitelist_and_greedy_ties_are_deterministic() {
        let processor = SlidingNoRepeatNgram::new(1, 4, [2]);
        let mut logits = vec![0.0, 3.0, 4.0, 3.0];
        processor.ban_in_place(&[1, 2, 3], &mut logits).unwrap();
        assert_eq!(greedy_token(&logits).unwrap(), 2);
        assert_eq!(greedy_token(&[1.0, 2.0, 2.0]).unwrap(), 1);
    }
}

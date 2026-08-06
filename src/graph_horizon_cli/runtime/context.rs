/*
 * Graph Horizon CLI context capacity model
 * Single responsibility: estimate Unicode text occupancy and enforce the
 * checked 90%-of-context admission budget without I/O or rendering.
 */

const CHARACTERS_PER_TOKEN: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContextBudget {
    context_limit: usize,
    max_tokens: usize,
    safe_total_budget: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContextUsage {
    pub(crate) estimated_messages: usize,
    pub(crate) context_limit: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CapacityError {
    pub(crate) estimated_messages: usize,
    pub(crate) max_tokens: usize,
    pub(crate) safe_total_budget: usize,
}

impl ContextBudget {
    pub(crate) fn new(context_limit: usize, max_tokens: usize) -> Option<Self> {
        let quotient = context_limit / 10;
        let remainder = context_limit % 10;
        // Quotient/remainder keeps floor(context_limit * 90 / 100) exact
        // without multiplying an untrusted context limit by 90.
        let safe_total_budget = quotient * 9 + remainder * 9 / 10;
        (context_limit > 0 && max_tokens < safe_total_budget).then_some(Self {
            context_limit,
            max_tokens,
            safe_total_budget,
        })
    }

    pub(crate) fn usage(self, characters: usize) -> ContextUsage {
        ContextUsage {
            estimated_messages: estimate_tokens(characters),
            context_limit: self.context_limit,
        }
    }

    pub(crate) fn admit(self, characters: usize) -> Result<ContextUsage, CapacityError> {
        let usage = self.usage(characters);
        let required = usage.estimated_messages.checked_add(self.max_tokens);
        if required.is_some_and(|required| required <= self.safe_total_budget) {
            Ok(usage)
        } else {
            Err(self.error(usage.estimated_messages))
        }
    }

    pub(crate) fn overflow_error(self) -> CapacityError {
        self.error(usize::MAX)
    }

    fn error(self, estimated_messages: usize) -> CapacityError {
        CapacityError {
            estimated_messages,
            max_tokens: self.max_tokens,
            safe_total_budget: self.safe_total_budget,
        }
    }
}

fn estimate_tokens(characters: usize) -> usize {
    characters / CHARACTERS_PER_TOKEN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_estimate_divides_summed_unicode_once() {
        let messages = ["abc", "😀éx"];
        let characters = messages.iter().map(|text| text.chars().count()).sum();
        let budget = ContextBudget::new(100, 1).unwrap();

        assert_eq!(budget.usage(characters).estimated_messages, 1);
    }

    #[test]
    fn safe_budget_is_exact_without_overflow() {
        let budget = ContextBudget::new(usize::MAX, 0).unwrap();
        let expected = usize::MAX / 10 * 9 + (usize::MAX % 10) * 9 / 10;

        assert_eq!(budget.safe_total_budget, expected);
    }

    #[test]
    fn budget_accepts_equality() {
        let budget = ContextBudget::new(90, 10).unwrap();
        assert_eq!(budget.admit(71 * 4).unwrap().estimated_messages, 71);
    }

    #[test]
    fn budget_rejects_overflow() {
        let safe = usize::MAX / 10 * 9 + (usize::MAX % 10) * 9 / 10;
        let budget = ContextBudget::new(usize::MAX, safe - 1).unwrap();

        assert_eq!(budget.admit(usize::MAX), Err(budget.error(usize::MAX / 4)));
    }

    #[test]
    fn reserve_must_leave_prompt_space() {
        assert!(ContextBudget::new(0, 0).is_none());
        assert!(ContextBudget::new(100, 90).is_none());
        assert!(ContextBudget::new(100, 89).is_some());
    }
}

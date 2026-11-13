# Analysis

## Layer 8, Head 11

This attention head appears to connect determiners (and adjectives) to the noun they introduce. Determiners such as "the", "a", "this" and "those" attend most strongly to the head noun of their noun phrase, even when an adjective sits between them (e.g. "a" -> "car" in "a red car"). Adjectives in the phrase tend to attend to the same noun.

Example Sentences:
- We saw the tall [MASK] near those trees. ("the" -> [MASK], "tall" -> [MASK], "those" -> "trees")
- I bought a red car yesterday [MASK]. ("a" -> "car", skipping "red")
- She read this old book in the [MASK]. ("this" -> "book")

## Layer 9, Head 11

This attention head appears to resolve pronouns to their antecedents. Pronouns such as "she", "his" and "they" attend most strongly to the noun they refer to earlier in the sentence, while most other tokens attend to [SEP]. This suggests the head plays a role in coreference.

Example Sentences:
- The doctor said she would call the [MASK]. ("she" -> "doctor")
- John lost his keys at the [MASK]. ("his" -> "john")
- The dogs barked because they were [MASK]. ("they" -> "dogs")

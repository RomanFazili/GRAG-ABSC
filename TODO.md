# TODO

- bm25 & simcse should also report back category
- bm25 & simcse should limit to k opinions, not sentences
- we need to adjust prompt to also take into account category

## Preprocessing

- Add a method to remove duplicate (and sometimes conflicting) opinions
    <Opinion target="food" category="FOOD#QUALITY" polarity="positive" from="5" to="9" />
    <Opinion target="food" category="FOOD#QUALITY" polarity="negative" from="5" to="9" />

    How do we handle these? Conflicting -> remove both. Just duplicate -> remove either.

## Models

- How do we ensure we stay within the models context window? What if we don't?
- First testrun on main.py gives a 52% accuracy. How did quinten et al. get 90% with default settings?
-> what changed?

## General bug fixing
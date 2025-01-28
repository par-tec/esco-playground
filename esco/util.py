"""
Aggregates skills and occupations data by `uri`,
generating text fields and label sets for semantic search.
"""

import pandas as pd


def is_valid(value):
    """
    Checks if the input value is valid.
    """
    if pd.isna(value):
        return False
    return bool(value)


def _aggregate_skills(df):
    """
    Aggregate skills by uri.
    Concatenate the values label, altLabel and description in the `text` column separated by "; "

    """
    skills = df.groupby("uri").agg(
        {
            "label": lambda x: x.iloc[0],
            "altLabel": lambda x: list(filter(is_valid, set(x))),
            "description": lambda x: x.iloc[0] if is_valid(x.iloc[0]) else "",
            "skillType": lambda x: x.iloc[0],
            "narrowers": lambda x: x.iloc[0],
        }
    )
    # Add a lowercase text field for semantic search.

    skills["text"] = skills.apply(
        lambda x: "; ".join(
            filter(
                is_valid,
                [x.label] + [label for label in x.altLabel if label] + [x.description],
            )
        ).lower(),
        axis=1,
    )
    # .. and a set of all the labels for each skill.

    skills["allLabel"] = skills.apply(
        lambda x: {t.lower() for t in x.altLabel} | {x.label.lower()}, axis=1
    )
    return skills


def _aggregate_occupations(df):
    """
    Aggregate occupations by uri.
    """
    o = df.groupby("uri").apply(
        lambda x: pd.Series(
            {
                "label": x.label.iloc[0],
                "altLabel": list(set(x.altLabel)),
                "description": x.description.iloc[0],
                "skill": list(set(x.skill.values)),
                "skill_": list(set(x[x.skillType.str.endswith("skill")].skill.values)),
                "knowledge_": list(
                    set(x[x.skillType.str.endswith("knowledge")].skill.values)
                ),
                "s": list(set(x.s.values)),
            }
        )
    )
    # Add a lowercase text field for semantic search.
    o["text"] = o.apply(
        lambda x: "; ".join([x.label] + x.altLabel + [x.description]).lower(), axis=1
    )
    # .. and a set of all the labels for each skill.

    o["allLabel"] = o.apply(
        lambda x: {t.lower() for t in x.altLabel} | {x.label.lower()}, axis=1
    )
    return o

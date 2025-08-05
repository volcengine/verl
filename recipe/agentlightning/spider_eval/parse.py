# Copyright (c) taoyds.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa
# type: ignore
# The evaluation code is from https://github.com/taoyds/test-suite-sql-eval

import itertools
import re
from collections import namedtuple
from typing import Any, Iterator

import sqlparse
from sqlparse.sql import Comparison, Identifier
from sqlparse.tokens import Whitespace

Token = namedtuple("Token", ["ttype", "value"])
VALUE_NUM_SYMBOL = "VALUERARE"
QUOTE_CHARS = {"`", "'", '"'}


def tokenize(query: str) -> list[Token]:
    tokens = list([Token(t.ttype, t.value) for t in sqlparse.parse(query)[0].flatten()])
    return tokens


def join_tokens(tokens: list[Token]) -> str:
    return "".join([x.value for x in tokens]).strip().replace("  ", " ")


def round_trip_test(query: str) -> None:
    tokens = tokenize(query)
    reconstructed = "".join([token.value for token in tokens])
    assert query == reconstructed, "Round trip test fails for string %s" % query


def postprocess(query: str) -> str:
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return query


# strip_query, reformat_query and replace values
# were implemented by Yu Tao for processing CoSQL
def strip_query(query: str) -> tuple[list[str], list[str]]:
    query_keywords, all_values = [], []

    # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}

    # Tao's implementation is commented out here.
    """
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)
    values = str_1 + str_2
        """

    toks = sqlparse.parse(query)[0].flatten()
    values = [
        t.value
        for t in toks
        if t.ttype == sqlparse.tokens.Literal.String.Single or t.ttype == sqlparse.tokens.Literal.String.Symbol
    ]

    for val in values:
        all_values.append(val)
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

    query_tokenized = query.split()
    float_nums = re.findall("[-+]?\d*\.\d+", query)
    all_values += [qt for qt in query_tokenized if qt in float_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]

    query = " ".join(query_tokenized)
    int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    all_values += [qt for qt in query_tokenized if qt in int_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    # print int_nums, query, query_tokenized

    for tok in query_tokenized:
        if "." in tok:
            table = re.findall("[Tt]\d+\.", tok)
            if len(table) > 0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t) > 0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())

        elif len(tok) > 0:
            query_keywords.append(tok.lower())
    return query_keywords, all_values


def reformat_query(query: str) -> str:
    query = query.strip().replace(";", "").replace("\t", "")
    query = " ".join([t.value for t in tokenize(query) if t.ttype != sqlparse.tokens.Whitespace])
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    return query


def replace_values(sql: str) -> tuple[list[str], set[str]]:
    sql = sqlparse.format(sql, reindent=False, keyword_case="upper")
    # sql = re.sub(r"(<=|>=|!=|=|<|>|,)", r" \1 ", sql)
    sql = re.sub(r"(T\d+\.)\s", r"\1", sql)
    query_toks_no_value, values = strip_query(sql)
    return query_toks_no_value, set(values)


# extract the non-value tokens and the set of values
# from a sql query
def extract_query_values(sql: str) -> tuple[list[str], set[str]]:
    reformated = reformat_query(query=sql)
    query_value_replaced, values = replace_values(reformated)
    return query_value_replaced, values


# plug in the values into query with value slots
def plugin(query_value_replaced: list[str], values_in_order: list[str]) -> str:
    q_length = len(query_value_replaced)
    query_w_values = query_value_replaced[:]
    value_idx = [idx for idx in range(q_length) if query_value_replaced[idx] == VALUE_NUM_SYMBOL.lower()]
    assert len(value_idx) == len(values_in_order)

    for idx, value in zip(value_idx, values_in_order, strict=False):
        query_w_values[idx] = value
    return " ".join(query_w_values)


# a generator generating all possible ways of
# filling values into predicted query
def plugin_all_permutations(query_value_replaced: list[str], values: set[str]) -> Iterator[str]:
    num_slots = len([v for v in query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    for values in itertools.product(*[list(values) for _ in range(num_slots)]):
        yield plugin(query_value_replaced, list(values))


# given the gold query and the model prediction
# extract values from the gold, extract predicted sql with value slots
# return 1) number of possible ways to plug in gold values and 2) an iterator of predictions with value plugged in
def get_all_preds_for_execution(gold: str, pred: str) -> tuple[int, Iterator[str]]:
    _, gold_values = extract_query_values(gold)
    pred_query_value_replaced, _ = extract_query_values(pred)
    num_slots = len([v for v in pred_query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    num_alternatives = len(gold_values) ** num_slots
    return num_alternatives, plugin_all_permutations(pred_query_value_replaced, gold_values)


def remove_distinct(s):
    toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
    return "".join([t for t in toks if t.lower() != "distinct"])


def extract_all_comparison_from_node(node: Token) -> list[Comparison]:
    comparison_list = []
    if hasattr(node, "tokens"):
        for t in node.tokens:
            comparison_list.extend(extract_all_comparison_from_node(t))
    if type(node) == Comparison:
        comparison_list.append(node)
    return comparison_list


def extract_all_comparison(query: str) -> list[Comparison]:
    tree = sqlparse.parse(query)[0]
    comparison_list = extract_all_comparison_from_node(tree)
    return comparison_list


def extract_toks_from_comparison(comparison_node: Comparison) -> list[Token]:
    tokens = [t for t in comparison_node.tokens if t.ttype != Whitespace]
    return tokens


def extract_info_from_comparison(comparison_node: Comparison) -> dict[str, Any]:
    tokens = extract_toks_from_comparison(comparison_node)
    left, op, right = tokens

    returned_dict = {"left": left, "op": op.value, "right": right}

    if type(left) != Identifier:
        return returned_dict

    table = None
    if len(left.tokens) == 3 and re.match("^[tT][0-9]$", left.tokens[0].value) is None:
        table = left.tokens[0].value.lower()
    col = left.tokens[-1].value

    if type(right) == Identifier:
        if len(right.tokens) == 1 and type(right.tokens[0]) == sqlparse.sql.Token:
            right_val = right.tokens[0].value
        else:
            return returned_dict
    elif type(right) == sqlparse.sql.Token:
        right_val = right.value
    else:
        return returned_dict

    returned_dict["table_col"], returned_dict["val"] = (table, col.upper()), process_str_value(right_val)

    return returned_dict


def extract_all_comparison_from_query(query: str) -> list[dict[str, Any]]:
    comparison_list = extract_all_comparison(query)
    return [extract_info_from_comparison(c) for c in comparison_list]


def extract_typed_value_in_comparison_from_query(query: str) -> list[tuple[tuple[str | None, str], str]]:
    cmps = extract_all_comparison_from_query(query)
    typed_values = [(cmp["table_col"], cmp["val"]) for cmp in cmps if "table_col" in cmp]
    for table, col, val1, val2 in re.findall(
        "(?:([^\.\s]*)\.)?([^\.\s]+) between ([^\s;]+) and ([^\s;]+)", query, re.IGNORECASE
    ):
        if table == "":
            table = None
        else:
            table = table.lower()
        col = col.upper()
        for v in [val1, val2]:
            typed_values.append(((table, col), v))
    return typed_values


def process_str_value(v: str) -> str:
    if len(v) > 0 and v[0] in QUOTE_CHARS:
        v = v[1:]
    if len(v) > 0 and v[-1] in QUOTE_CHARS:
        v = v[:-1]
    for c in QUOTE_CHARS:
        v = v.replace(c + c, c)
    return v

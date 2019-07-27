# -*- coding: utf-8 -*-

import pytest
from pytest_gherkin import action


@action("I have <n> <fruit>")
def have(fruit, n: int, basket):
    assert n >= 0
    basket[fruit] = n


@action("I eat <m> <fruit>")
def eat(m: int, fruit, basket):
    assert m >= 0
    basket[fruit] -= m
    assert basket[fruit] >= 0


@action("I have <m> <stuff> remaining")
def have_remaining(m: int, stuff, basket):
    assert basket[stuff] == m


@action("I have <x> things remaining")
def have_things_remaining(x: int, basket):
    assert sum(basket.values()) == x

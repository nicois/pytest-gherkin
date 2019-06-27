Feature: name of a fruity feature
    As a person
    I want to eat some fruit
    So that I grow strong

    Background:
        Given I have <17> <tomatoes>

    Scenario Outline: eat fruit and vegies
        Given I have <foo> <fruit>
        And I have <foo> <vegetable>
        When I eat <bar> <fruit>
        And I eat <bar> <vegetable>
        Then I have <baz> <fruit> remaining
        And I have <baz> <vegetable> remaining
        And I have <17> <tomatoes> remaining
        And I have <total> things remaining

        Examples:
            | fruit   | vegetable | foo | bar | baz | total |
            | apples  | cucumber  | 10  | 4   |   6 |  29   |
            | oranges | broccoli  | 8   | 3   |   5 |  27   |


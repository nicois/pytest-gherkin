Feature: galaxy
    Scenario Outline: discovery
        Given There are <A> solar systems
        When I discover another solar system
        And I discover another solar system
        And I discover another solar system
        Then I have <B> solar systems

        Examples:
            | A  | B  |
            | 10 | 13 |
            | 0  | 3  |


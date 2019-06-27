Feature: patients have appointments
    Scenario Outline: count appointments
        Given a patient exists
        And the patient has <number> appointments in the future
        Then five appointments exist

        Examples:
            | number |
            | 5      |


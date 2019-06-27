Feature: patients have appointments
    Scenario: count appointments no outline
        Given a patient exists
        And the patient has <5> appointments in the future
        Then five appointments exist

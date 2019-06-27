from pytest_gherkin.legacy import given, parse, then


@given("a patient exists", target_fixture="patient")
def a_patient():
    return dict(patient=True)


@given("a patient with <attr> equal to <value> exists", target_fixture="patient")
def create_custom_attribute_patient(attr, value):
    patient = dict(patient=True)
    patient[attr] = value
    return patient


@given(
    parse("the patient has {number} appointments in the future"),
    target_fixture="appointments_for_patient",
)
def generate_appointments(patient, number):
    return [patient] * int(number)


@then("five appointments exist")
def five(appointments_for_patient):
    assert len(appointments_for_patient) == 5

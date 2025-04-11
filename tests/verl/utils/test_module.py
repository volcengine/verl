# Test module for import_utils.load_extern_type testing
class TestClass:
    """A test class to be imported by load_extern_type"""

    def __init__(self, value=None):
        self.value = value or "default"

    def get_value(self):
        return self.value


TEST_CONSTANT = "test_constant_value"


def test_function():
    return "test_function_result"

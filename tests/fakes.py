"""Shared test fakes used across multiple test files."""


class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeContent(text)]


class FakeMessages:
    def __init__(self):
        self.responses = []
        self.call_count = 0

    def add_response(self, text):
        self.responses.append(text)

    def create(self, **kwargs):
        response = FakeResponse(self.responses[self.call_count])
        self.call_count += 1
        return response


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = FakeMessages()

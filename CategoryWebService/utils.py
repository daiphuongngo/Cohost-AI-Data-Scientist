import json


class IntentClass:
    def __init__(self, name, confident):
        self.name = name
        self.confident = confident


class CategoryClass:
    def __init__(self, name, confident):
        self.name = name
        self.confident = confident


class ResponseClass:
    def __init__(self, text, intent_obj, category_obj):
        self.text = text
        self.intent = intent_obj
        self.category = category_obj

    def toJSON(self):
        return json.loads(json.dumps(self, default=lambda o: o.__dict__,
                                     sort_keys=True, indent=4))

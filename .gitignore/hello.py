import json

# print("Helll")

import json

class Aclasss:
    def __init__(self):
        self.a = "Hello"
        self.b = "Hellwoudl"
    # def toJSON(self):
    #     return json.dumps(self, default=lambda o: o.__dict__,
    #         sort_keys=True, indent=4)




class Object:
    def __init__(self):
        self.a = "Hello"
        self.b = Aclasss()
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

class IntentClass:
    def __init__(self, name, confident):
        self.name = name
        self.confident = confident

class CategoryClass:
    def __init__(self, name, confident):
        self.name = name
        self.confident = confident


class ResponseClass:
    def __init__(self, text, intentObj, categoryObj):
        self.text = text
        self.intent = intentObj
        self.category = categoryObj

    def toJSON(self):
        return json.loads(json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4))

intentObj = IntentClass("aadada", 0.00001111)
categoryObj = CategoryClass("aadada", 0.00001111)
obj = Object()
print(ResponseClass("sssss", intentObj, categoryObj).toJSON())
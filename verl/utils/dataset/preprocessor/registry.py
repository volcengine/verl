class Register:
    def __init__(self, name):
        self._name = name
        self._obj_dict = dict()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, keys={list(self._obj_dict.keys())})"

    def _register(self, name, obj):
        if name in self._obj_dict:
            raise KeyError(f"Object '{name}' is already registered in '{self._name}'")
        self._obj_dict[name.lower()] = obj

    def register(self, obj=None):
        if obj is None: # Used as a decorator
            def deco(func_or_class):
                self._register(func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        # Used as a function call
        self._register(obj.__name__, obj)
        return obj

    def get(self, name):
        obj = self._obj_dict.get(name.lower())
        if obj is None:
            raise KeyError(f"Object '{name}' not found in '{self._name}' registry.")
        return obj

PREPROCESSOR_REGISTER = Register("Preprocessor Register")
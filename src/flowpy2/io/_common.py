import os
from importlib import import_module

valid_tag_checks = ("strict", "weak",'warn', 'nocheck')

def make_tag(cls) -> str:
    return '.'.join([cls.__module__,
                     cls.__name__])

def weak_tag_check(cls: type, tag: str, error: type) -> bool:
    module_name, class_name = os.path.splitext(tag)
    try:
        module = import_module(module_name)
        tag_class =  getattr(module,class_name[1:])
    except ModuleNotFoundError:
        raise error("Tag check points to module "
                    "that doesn't exist") from None
    except AttributeError:
        raise error("Tag check points to class that"
                    " doesn't exist") from None
    
    if tag_class not in cls.mro():
        raise error(f"Class {cls.__name__} is not a "
                    f"subclass of {tag}. You can change "
                    "tag check through keyword or rcParams")
import os
from importlib import import_module

valid_tag_checks = ("strict", "weak",'warn', 'nocheck')

def make_tag(cls) -> str:
    return '.'.join([cls.__module__,
                     cls.__name__])

def cls_from_tag(tag, error=None):
    module_name, class_name = os.path.splitext(tag)
    try:
        module = import_module(module_name)
        return  getattr(module,class_name[1:])
    except ModuleNotFoundError as e:
        if error is not None:
            raise error("Tag check points to module "
                        "that doesn't exist") from None
        
        raise e from None
    
    except AttributeError as e:
        if error is not None:
            raise error("Tag check points to class that"
                        " doesn't exist") from None
        raise e from None
    
def weak_tag_check(cls: type, tag: str, error: type) -> bool:
    
    tag_class = cls_from_tag(tag, error=error)

    if not issubclass(cls, tag_class) and not issubclass(tag_class, cls):
        raise error(f"Class {cls.__name__} is not a "
                    f"subclass of {tag} or vice versa. You can change "
                    "tag check through keyword or rcParams")
    return tag_class